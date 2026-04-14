"""LangGraph-powered orchestration for recommendation and follow-up flows."""

from __future__ import annotations

import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast

from dotenv import dotenv_values
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.logging_utils import get_logger, summarize_text
from src.rag.prompts import (
    get_follow_up_background_system_prompt,
    get_follow_up_synthesis_system_prompt,
    get_query_recognition_system_prompt,
)
from src.rag.rag_service import RAGService


logger = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"

DEFAULT_SCOPE_MESSAGE = (
    "As a book recommendation assistant, I can only help with suggesting books or discussing books of my catalogue. "
    "Let me know what themes interest you and I will suggest some relevant books."
)

DEFAULT_PROBLEMATIC_MESSAGE = (
    "As a book recommendation assistant, I can only help with suggesting books or discussing books of my catalogue. "
    "Let me know what themes interest you and I will suggest some relevant books."
)

DEFAULT_CLARIFICATION_MESSAGE = (
    "Do you want a new book recommendation, or do you want more information about "
    "one of the books or authors I already suggested?"
)

FOLLOW_UP_INVITATION = (
    "If one of these books interests you, you can ask me more about the book or its author."
)


class QueryRoute(str, Enum):
    """Available classifier routes."""

    IRRELEVANT = "irrelevant"
    PROBLEMATIC = "problematic"
    NEW_RECOMMENDATION = "new_recommendation"
    FOLLOW_UP = "follow_up"
    AMBIGUOUS = "ambiguous"


class QueryRecognitionResult(BaseModel):
    """Structured output for query recognition."""

    route: QueryRoute
    retrieval_query: str = Field(default="")
    referenced_items: list[str] = Field(default_factory=list)
    reason: str = Field(default="")


class ThreadSnapshot(TypedDict, total=False):
    """In-memory thread state."""

    messages: list[dict[str, str]]
    recommended_books: list[dict[str, Any]]
    recommended_books_summary: str


class AgenticGraphState(TypedDict, total=False):
    """LangGraph state for a single turn."""

    thread_id: str
    user_query: str
    has_history: bool
    messages: list[dict[str, str]]
    recommended_books: list[dict[str, Any]]
    recommended_books_summary: str
    route: str
    retrieval_query: str
    referenced_items: list[str]
    response: str
    recommendations: list[dict[str, Any]]
    sources: list[dict[str, str]]
    follow_up_search_query: str
    tavily_result: dict[str, Any]
    wikipedia_result: dict[str, Any]
    llm_background: str


class AgenticRAGService:
    """Route user turns through recommendation or follow-up workflows."""

    def __init__(
        self,
        llm: ChatOpenAI,
        vectorstore: Any,
        k: int = 5,
        classifier_llm: ChatOpenAI | None = None,
        tavily_max_results: int = 5,
        wikipedia_top_k_results: int = 3,
    ) -> None:
        """ 
        Initialize the service with necessary components and parameters.
        Args:
            llm: The language model to use for generating responses.
            vectorstore: The vector store for retrieving relevant documents.
            k: The number of top documents to retrieve.
            classifier_llm: Optional language model for query classification.
            tavily_max_results: Maximum number of results from Tavily search.
            wikipedia_top_k_results: Maximum number of results from Wikipedia search.
        """
        if llm is None:
            raise ValueError("llm must not be None")
        if vectorstore is None:
            raise ValueError("vectorstore must not be None")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        self.llm = llm
        self.classifier_llm = classifier_llm or llm
        self.recommendation_service = RAGService(llm=llm, vectorstore=vectorstore, k=k)
        self.tavily_max_results = tavily_max_results
        self.wikipedia_top_k_results = wikipedia_top_k_results
        self._threads: dict[str, ThreadSnapshot] = {}
        self.graph = self._build_graph()
        logger.info("Initialized agentic RAG service")

    def _get_dotenv_value(self, key: str) -> str | None:
        """
        Read a value from the project .env file.
        Args:
            key: The environment variable name to read.
        Returns:
            The string value if present and non-empty, otherwise None.
        """
        value = dotenv_values(_DOTENV_PATH).get(key)
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def answer_query(self, query: str, thread_id: str) -> dict[str, Any]:
        """
        Route and answer the given user query for a specific thread.

        Args:
            query: The user's query as a string.
            thread_id: The identifier for the conversation thread.

        Returns:
            A dictionary containing the response, recommendations, and sources.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Received invalid query: query must be a non-empty string")
        if not thread_id or not isinstance(thread_id, str):
            raise ValueError("Received invalid thread_id: thread_id must be a non-empty string")

        # Load the current memory snapshot for the thread, if it exists
        snapshot = self._threads.get(thread_id, {})
        logger.info(
            "Running agentic workflow for thread '%s' and query '%s'",
            thread_id,
            summarize_text(query),
        )

        # Invoke the graph with the current state and get the result
        result = self.graph.invoke(
            {
                "thread_id": thread_id,
                "user_query": query,
                "messages": list(snapshot.get("messages", [])),
                "recommended_books": list(snapshot.get("recommended_books", [])),
                "recommended_books_summary": str(snapshot.get("recommended_books_summary", "")),
                "has_history": bool(snapshot.get("recommended_books")),
                "recommendations": [],
                "sources": [],
            }
        )

        # Persist the updated thread state based on the graph's output
        self._persist_thread_state(
            thread_id=thread_id,
            user_query=query,
            response=str(result.get("response", "")),
            recommendations=cast(list[dict[str, Any]], result.get("recommendations", [])),
            previous_messages=list(snapshot.get("messages", [])),
            previous_summary=str(
                result.get("recommended_books_summary", snapshot.get("recommended_books_summary", ""))
            ),
            route=str(result.get("route", "")),
        )

        # Return the response, recommendations, and sources to the caller
        return {
            "response": result.get("response", ""),
            "recommendations": result.get("recommendations", []),
            "sources": result.get("sources", []),
        }

    def _build_graph(self)-> StateGraph[AgenticGraphState]:
        """
        Construct the LangGraph state machine for routing queries.

        Returns:
            A StateGraph instance representing the routing logic.
        """
        # Build the nodes
        builder = StateGraph(AgenticGraphState)
        builder.add_node("recognize_query", self._recognize_query_node)
        builder.add_node("reject_query", self._reject_query_node)
        builder.add_node("ask_clarification", self._ask_clarification_node)
        builder.add_node("recommend_books", self._recommend_books_node)
        builder.add_node("prepare_follow_up", self._prepare_follow_up_node)
        builder.add_node("tavily_lookup", self._tavily_lookup_node)
        builder.add_node("wikipedia_lookup", self._wikipedia_lookup_node)
        builder.add_node("llm_background_lookup", self._llm_background_lookup_node)
        builder.add_node("answer_follow_up", self._answer_follow_up_node)

        # Build the edges
        builder.add_edge(START, "recognize_query")
        builder.add_conditional_edges(
            "recognize_query",
            self._route_from_state,
            {
                QueryRoute.IRRELEVANT.value: "reject_query",
                QueryRoute.PROBLEMATIC.value: "reject_query",
                QueryRoute.AMBIGUOUS.value: "ask_clarification",
                QueryRoute.NEW_RECOMMENDATION.value: "recommend_books",
                QueryRoute.FOLLOW_UP.value: "prepare_follow_up",
            },
        )
        builder.add_edge("reject_query", END)
        builder.add_edge("ask_clarification", END)
        builder.add_edge("recommend_books", END)
        builder.add_edge("prepare_follow_up", "tavily_lookup")
        builder.add_edge("prepare_follow_up", "wikipedia_lookup")
        builder.add_edge("prepare_follow_up", "llm_background_lookup")
        builder.add_edge(
            ["tavily_lookup", "wikipedia_lookup", "llm_background_lookup"],
            "answer_follow_up",
        )
        builder.add_edge("answer_follow_up", END)
        return builder.compile()

    def _route_from_state(self, state: AgenticGraphState) -> str:
        """
        Determine the routing path based on the current state after query recognition.
        Args:
            state: The current state of the graph after processing the query recognition node.
        Returns:
            A string representing the route to take, which should match one of the QueryRoute values.
        """
        route = str(state.get("route", QueryRoute.IRRELEVANT.value))
        logger.info("Routing agentic workflow to '%s'", route)
        return route

    def _recognize_query_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Classify the user's query to determine the appropriate route for handling it.
        Args:
            state: The current state containing the user's query and context.
        Returns:
            A dictionary with the classification results, including the route, retrieval query, and referenced items.
        """
        logger.info(
            "Running query recognition node for query '%s' with history=%s",
            summarize_text(str(state["user_query"])),
            bool(state.get("has_history")),
        )
        # Classify the user query
        classification = self._recognize_query(
            user_query=str(state["user_query"]),
            has_history=bool(state.get("has_history")),
            recommended_books=cast(list[dict[str, Any]], state.get("recommended_books", [])),
            messages=cast(list[dict[str, str]], state.get("messages", [])),
            recommended_books_summary=str(state.get("recommended_books_summary", "")),
        )
        logger.info(
            "Query recognition selected route='%s', retrieval_query='%s', referenced_items=%s",
            classification.route.value,
            summarize_text(classification.retrieval_query or str(state["user_query"])),
            classification.referenced_items,
        )
        # Update the state
        next_state: dict[str, Any] = {
            "route": classification.route.value,
            "retrieval_query": classification.retrieval_query.strip() or str(state["user_query"]).strip(),
            "referenced_items": classification.referenced_items,
        }
        # If the user is asking for a new recommendation and there is existing history, reset the recommended books context
        if classification.route == QueryRoute.NEW_RECOMMENDATION and state.get("has_history"):
            next_state["recommended_books_summary"] = self._build_reset_summary(state)
            next_state["recommended_books"] = []
        return next_state

    def _reject_query_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Handle the rejection of a query based on its classification.
        Args:
            state: The current state containing the user's query and context.
        Returns:
            A dictionary with the response message, recommendations, and sources.
        """
        # Determine the appropriate message based on the route
        route = str(state.get("route", QueryRoute.IRRELEVANT.value))
        logger.info("Reject query node handling route='%s'", route)
        message = DEFAULT_SCOPE_MESSAGE
        if route == QueryRoute.PROBLEMATIC.value:
            message = DEFAULT_PROBLEMATIC_MESSAGE

        # Return the rejection response without any recommendations or sources
        return {
            "response": message,
            "recommendations": [],
            "sources": [],
        }

    def _ask_clarification_node(self, _state: AgenticGraphState) -> dict[str, Any]:
        """
        Handle the scenario where the system needs to ask the user for clarification.
        Args:
            state: The current state containing the user's query and context.
        Returns:
            A dictionary with the clarification message, recommendations, and sources.
        """
        logger.info("Clarification node selected")
        # Ask for clarification without providing any recommendations or sources
        return {
            "response": DEFAULT_CLARIFICATION_MESSAGE,
            "recommendations": [],
            "sources": [],
        }

    def _recommend_books_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Handle the scenario where the user is asking for new book recommendations.
        Args:
            state: The current state containing the user's query and context.
        Returns:
            A dictionary with the response message, list of recommended books, and sources.
        """
        # Get the query
        query = str(state.get("retrieval_query") or state["user_query"]).strip()
        logger.info("Recommendation node retrieving books for query '%s'", summarize_text(query))

        # Call the RAG service
        result = self.recommendation_service.answer_query(query)
        response = str(result.get("response", "")).strip()
        recommendations = cast(list[dict[str, Any]], result.get("recommendations", []))
        logger.info("Recommendation node produced %d recommendation(s)", len(recommendations))

        # Add the follow-up invitation if there are any recommendations to discuss
        if recommendations:
            response = f"{response}\n\n{FOLLOW_UP_INVITATION}".strip()
        return {
            "response": response,
            "recommendations": recommendations,
            "sources": [],
            "recommended_books": recommendations,
        }

    def _answer_follow_up_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Handle the scenario where the user is asking a follow-up question about previously recommended books.
        Args:
            state: The current state containing the user's query and context.
        Returns:
            A dictionary with the response message, recommendations, and sources.
        """
        logger.info("Follow-up synthesis node combining tool outputs")
        # Get outside context and merge it
        tavily_result = cast(dict[str, Any], state.get("tavily_result", {}))
        wikipedia_result = cast(dict[str, Any], state.get("wikipedia_result", {}))
        llm_background = str(state.get("llm_background", ""))
        sources = self._merge_sources(
            cast(list[dict[str, Any]], tavily_result.get("sources", [])),
            cast(list[dict[str, Any]], wikipedia_result.get("sources", [])),
        )
        logger.info(
            "Follow-up synthesis received tavily_summary=%s wikipedia_summary=%s llm_background=%s sources=%d",
            bool(str(tavily_result.get("summary", "")).strip()),
            bool(str(wikipedia_result.get("summary", "")).strip()),
            bool(llm_background.strip()),
            len(sources),
        )
        # Generate the follow-up answer
        response = self._synthesize_follow_up_answer(
            user_query=str(state["user_query"]),
            recommended_books=cast(list[dict[str, Any]], state.get("recommended_books", [])),
            messages=cast(list[dict[str, str]], state.get("messages", [])),
            tavily_summary=str(tavily_result.get("summary", "")),
            wikipedia_summary=str(wikipedia_result.get("summary", "")),
            llm_background=llm_background,
            sources=sources,
        )
        return {
            "response": response,
            "recommendations": [],
            "sources": sources,
        }

    def _prepare_follow_up_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Prepare the common search query used by the follow-up tool nodes.
        Args:
            state: The current state containing the user's query and context.
        Returns:
            A dictionary containing the follow-up search query.
        """
        follow_up_search_query = self._build_follow_up_search_query(state)
        logger.info(
            "Prepare follow-up node built search query '%s'",
            summarize_text(follow_up_search_query),
        )
        return {
            "follow_up_search_query": follow_up_search_query,
        }

    def _tavily_lookup_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Run Tavily search for the follow-up branch.
        Args:
            state: The current state containing the prepared follow-up search query.
        Returns:
            A dictionary containing the Tavily search result payload.
        """
        logger.info(
            "Tavily lookup node triggered for query '%s'",
            summarize_text(str(state.get("follow_up_search_query", ""))),
        )
        return {
            "tavily_result": self._run_tavily_search(str(state.get("follow_up_search_query", ""))),
        }

    def _wikipedia_lookup_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Run Wikipedia search for the follow-up branch.
        Args:
            state: The current state containing the prepared follow-up search query.
        Returns:
            A dictionary containing the Wikipedia search result payload.
        """
        logger.info(
            "Wikipedia lookup node triggered for query '%s'",
            summarize_text(str(state.get("follow_up_search_query", ""))),
        )
        return {
            "wikipedia_result": self._run_wikipedia_search(str(state.get("follow_up_search_query", ""))),
        }

    def _llm_background_lookup_node(self, state: AgenticGraphState) -> dict[str, Any]:
        """
        Run the LLM background lookup for the follow-up branch.
        Args:
            state: The current state containing the user's query and recommended books context.
        Returns:
            A dictionary containing the LLM background note.
        """
        logger.info(
            "LLM background lookup node triggered for query '%s'",
            summarize_text(str(state["user_query"])),
        )
        return {
            "llm_background": self._run_llm_background_lookup(
                str(state["user_query"]),
                cast(list[dict[str, Any]], state.get("recommended_books", [])),
                cast(list[dict[str, str]], state.get("messages", [])),
            ),
        }

    def _recognize_query(
        self,
        user_query: str,
        has_history: bool,
        recommended_books: list[dict[str, Any]],
        messages: list[dict[str, str]],
        recommended_books_summary: str,
    ) -> QueryRecognitionResult:
        """
        Classify the user query to determine if it's asking for a new recommendation, a follow-up about existing recommendations, or is irrelevant/problematic.
        Args:
            user_query: The raw query input from the user.
            has_history: A boolean indicating if there is existing recommendation history for the thread.
            recommended_books: The list of currently recommended books in the thread context.
            recommended_books_summary: A summary of the recommended books to provide additional context for classification.
        Returns:
            A QueryRecognitionResult object containing the classification route, any retrieval query, referenced items, and reasoning for the classification.
        """
        # get the titles of the recommended books
        titles = [
            f"{item.get('title', 'Unknown title')} by {item.get('author', 'Unknown author')}"
            for item in recommended_books
        ]
        # Create the classification prompt with the user query and context, and invoke the classifier LLM
        system_prompt = get_query_recognition_system_prompt(
            has_history=has_history,
            recommended_books=titles,
            recent_messages=self._format_recent_messages(messages),
            recommended_books_summary=recommended_books_summary,
        )
        llm = self.classifier_llm.with_structured_output(QueryRecognitionResult)
        return cast(
            QueryRecognitionResult,
            llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"User query: {user_query}"),
                ]
            ),
        )

    def _build_reset_summary(self, state: AgenticGraphState) -> str:
        """
        Build a summary message to capture the context of previous recommendations when a new recommendation request is made, which resets the thread's recommended books context. 
        Args:
            state: The current state containing the previous recommended books and their summary.
        Returns:
            A string summarizing the previous recommendations.
        """
        # Get the books and extract their titles and authors for the reset summary
        books = cast(list[dict[str, Any]], state.get("recommended_books", []))
        titles = [
            f"{item.get('title', 'Unknown title')} by {item.get('author', 'Unknown author')}"
            for item in books
        ]
        # Return the reset summary with the recommended books
        if titles:
            return (
                "Previous recommendation thread reset after a new request. Prior suggestions were: "
                + "; ".join(titles[:3])
                + "."
            )
        # If there are no books but there is a summary, return the existing summary to preserve any contextual information it may contain
        existing = str(state.get("recommended_books_summary", "")).strip()
        if existing:
            return existing
        return "Previous recommendation thread reset after a new request."

    def _build_follow_up_search_query(self, state: AgenticGraphState) -> str:
        """
        Build a search query for follow-up recommendations based on the user's query and the context of previously recommended books.
        Args:
            state: The current state containing the user's query, referenced items, and recommended books.
        Returns:
            A string representing the search query for follow-up recommendations.
        """
        # If the query references specific items, prioritize those in the search query
        referenced = cast(list[str], state.get("referenced_items", []))
        if referenced:
            return f"{state['user_query']} {' '.join(referenced)}"

        recent_messages = self._format_recent_messages(cast(list[dict[str, str]], state.get("messages", [])))
        if recent_messages:
            return f"{state['user_query']} {' '.join(recent_messages[-2:])}"
        
        # If not, combine all the recommended books into the search query to provide context for the follow-up question
        books = cast(list[dict[str, Any]], state.get("recommended_books", []))
        anchors = [
            f"{item.get('title', '')} {item.get('author', '')}".strip()
            for item in books[:3]
        ]
        joined = " | ".join(item for item in anchors if item)
        if joined:
            return f"{state['user_query']} {joined}"
        return str(state["user_query"])

    def _run_tavily_search(self, query: str) -> dict[str, Any]:
        """
        Perform a search using the Tavily API and return the results.
        Args:
            query: The search query string.
        Returns:
            A dictionary containing the summary and sources of the search results.
        """
        # If no API key accessible, return empty object without attempting the search
        if not self._get_dotenv_value("TAVILY_API_KEY"):
            logger.info("Skipping Tavily search because TAVILY_API_KEY is not configured")
            return {"summary": "", "sources": []}
        try:
            logger.info("Running Tavily search for query '%s'", summarize_text(query))
            tool = TavilySearch(max_results=self.tavily_max_results)
            raw_result = tool.invoke(query)
        except Exception as exc:
            logger.warning("Tavily search failed for '%s': %s", summarize_text(query), exc)
            return {"summary": "", "sources": []}

        # Parse the raw result to extract the response and the sources
        if isinstance(raw_result, tuple):
            content, artifact = raw_result
            summary = str(content)
            results = artifact if isinstance(artifact, dict) else {}
        elif isinstance(raw_result, dict):
            summary = str(raw_result.get("content", raw_result.get("answer", raw_result)))
            results = raw_result
        elif isinstance(raw_result, list):
            summary = "\n".join(str(item.get("content", item)) for item in raw_result if item)
            results = {"results": raw_result}
        else:
            summary = str(raw_result)
            results = {}

        # List all the sources in the results
        source_entries = []
        raw_sources = results.get("results", []) if isinstance(results, dict) else []
        if isinstance(raw_sources, list):
            for item in raw_sources:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                title = str(item.get("title", url or "Tavily result")).strip()
                if not url:
                    continue
                source_entries.append({"title": title, "url": url})
        logger.info("Tavily search returned %d source(s)", len(source_entries))
        return {"summary": summary, "sources": source_entries}

    def _run_wikipedia_search(self, query: str) -> dict[str, Any]:
        """
        Run the Wikipedia search for the follow-up branch and return the summary and sources.
        Args:
            query: The search query string.
        Returns:
            A dictionary containing the summary and sources of the Wikipedia search results.
        """
        # Run the Wikipedia search
        try:
            logger.info("Running Wikipedia search for query '%s'", summarize_text(query))
            wrapper = WikipediaAPIWrapper(top_k_results=self.wikipedia_top_k_results)
            summary = wrapper.run(query)
        except Exception as exc:
            logger.warning("Wikipedia search failed for '%s': %s", summarize_text(query), exc)
            return {"summary": "", "sources": []}

        if not summary:
            logger.info("Wikipedia search returned no summary")
            return {"summary": "", "sources": []}
        
        # Get the source URL
        source_url = (
            "https://en.wikipedia.org/wiki/Special:Search?search="
            + urllib.parse.quote_plus(query)
        )
        return {
            "summary": str(summary),
            "sources": [{"title": f"Wikipedia search: {query}", "url": source_url}],
        }

    def _run_llm_background_lookup(
        self,
        user_query: str,
        recommended_books: list[dict[str, Any]],
        messages: list[dict[str, str]],
    ) -> str:
        """
        Run a background lookup using the LLM to provide additional context for follow-up questions about previously recommended books or authors.
        Args:
            user_query: The original user query that may be asking a follow-up question.
            recommended_books: The list of currently recommended books in the thread context, which may be relevant for answering the follow-up question.
        Returns:
            A string containing the background note generated by the LLM, which may help answer the user's follow-up question.
        """
        # Get recommended books information
        titles = [
            f"{item.get('title', 'Unknown title')} by {item.get('author', 'Unknown author')}"
            for item in recommended_books
        ]
        logger.info(
            "Running LLM background lookup with %d recommended book anchor(s)",
            len(titles),
        )
        # Create the prompt to send to the LLM
        prompt = [
            SystemMessage(
                content=get_follow_up_background_system_prompt()
            ),
            HumanMessage(
                content=(
                    f"User question: {user_query}\n"
                    f"Previously recommended books: {titles if titles else 'None'}\n"
                    f"Recent conversation: {self._format_recent_messages(messages) or 'None'}"
                )
            ),
        ]
        # Get the LLM's response and return it as the background note
        result = self.llm.invoke(prompt)
        logger.info("LLM background lookup completed")
        return cast(str, result.content)

    def _synthesize_follow_up_answer(
        self,
        user_query: str,
        recommended_books: list[dict[str, Any]],
        messages: list[dict[str, str]],
        tavily_summary: str,
        wikipedia_summary: str,
        llm_background: str,
        sources: list[dict[str, str]],
    ) -> str:
        """
        Synthesize an answer to the user based on the Tavily, Wikipedia, and LLM background outputs.
        Args:
            user_query: The original user query that may be asking a follow-up question.
            recommended_books: The list of currently recommended books in the thread context, which may be relevant for answering the follow-up question.
            tavily_summary: The summary generated by Tavily for the current thread context.
            wikipedia_summary: The summary generated by Wikipedia for the current thread context.
            llm_background: The background note generated by the LLM for the current thread context.
            sources: The list of sources used to generate the summaries.
        Returns:
            A string containing the synthesized answer to the user's follow-up question.

        """
        # Get books information
        titles = [
            f"{item.get('title', 'Unknown title')} by {item.get('author', 'Unknown author')}"
            for item in recommended_books
        ]
        logger.info(
            "Running follow-up synthesis with %d recommended book anchor(s) and %d source(s)",
            len(titles),
            len(sources),
        )

        # Build the prompt
        prompt = [
            SystemMessage(
                content=get_follow_up_synthesis_system_prompt(
                    has_sources=bool(sources),
                )
            ),
            HumanMessage(
                content=(
                    f"User question: {user_query}\n"
                    f"Previously recommended books: {titles if titles else 'None'}\n"
                    f"Recent conversation: {self._format_recent_messages(messages) or 'None'}\n"
                    f"Tavily evidence:\n{tavily_summary or 'None'}\n\n"
                    f"Wikipedia evidence:\n{wikipedia_summary or 'None'}\n\n"
                    f"Background note:\n{llm_background or 'None'}\n\n"
                    f"Available sources: {sources if sources else 'None'}"
                )
            ),
        ]

        # Get the synthesized answer from the LLM
        result = self.llm.invoke(prompt)
        logger.info("Follow-up synthesis completed")
        return cast(str, result.content)

    def _merge_sources(
        self,
        tavily_sources: list[dict[str, Any]],
        wikipedia_sources: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """
        Merge and deduplicate sources from Tavily and Wikipedia to provide a unified list of sources for the follow-up answer.
        Args:
            tavily_sources: The list of sources returned from the Tavily search.
            wikipedia_sources: The list of sources returned from the Wikipedia search.
        Returns:
            A merged and deduplicated list of sources, where each source is a dictionary containing a title and a URL.
        """
        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        # For each source from both Tavily and Wikipedia, extract the title and URL, and add it to the merged list if it hasn't been seen before
        for item in [*tavily_sources, *wikipedia_sources]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", url or "Source")).strip()
            if not url or url in seen:
                continue
            merged.append({"title": title, "url": url})
            seen.add(url)
        return merged

    def _persist_thread_state(
        self,
        thread_id: str,
        user_query: str,
        response: str,
        recommendations: list[dict[str, Any]],
        previous_messages: list[dict[str, str]],
        previous_summary: str,
        route: str,
    ) -> None:
        """
        Persist the state of a conversation thread, including the user's query, the assistant's response, the list of recommended books, and a summary of the recommendations for future context. 
        This method updates the in-memory thread state based on the latest interaction and the route taken (new recommendation or follow-up).
        Args:
            thread_id: The identifier for the conversation thread.
            user_query: The user's query that was just processed.
            response: The assistant's response to the user's query.
            recommendations: The list of book recommendations generated in this turn, if any.
            previous_messages: The list of messages from the previous state of the thread, which may be used to maintain context in the conversation history.
            previous_summary: The summary of previously recommended books before this turn, which may be updated based on the route taken.
            route: The classification route taken for this query, which determines how the thread state should be updated (e.g., whether to reset recommendations or append to history).
        """
        messages = list(previous_messages)
        messages.append({"role": "user", "content": user_query})
        messages.append({"role": "assistant", "content": response})
        summary = previous_summary
        # If NEW_RECOMMENDATION route, build a new summary based on the new recommendations and either append it to the existing summary or replace it if there is no existing summary.
        if route == QueryRoute.NEW_RECOMMENDATION.value:
            new_summary = self._build_summary_from_recommendations(recommendations)
            summary = f"{summary} {new_summary}".strip() if summary and new_summary else (new_summary or summary)
        # If FOLLOW_UP route but no existing summary, build a new summary based on the recommendations to provide context for future follow-up questions.
        elif route == QueryRoute.FOLLOW_UP.value and not summary:
            summary = self._build_summary_from_recommendations(recommendations)
        logger.info(
            "Persisting thread '%s' with route='%s', messages=%d, recommended_books=%d",
            thread_id,
            route,
            len(messages[-20:]),
            len(recommendations or self._threads.get(thread_id, {}).get("recommended_books", [])),
        )
        self._threads[thread_id] = {
            "messages": messages[-20:],
            "recommended_books": recommendations or self._threads.get(thread_id, {}).get("recommended_books", []),
            "recommended_books_summary": summary,
        }

    def _build_summary_from_recommendations(self, recommendations: list[dict[str, Any]]) -> str:
        """
        Build a summary string from a list of book recommendations.
        Args:
            recommendations: A list of dictionaries, each containing information about a recommended book.
        Returns:
            A string summarizing the most recent recommended books.
        """
        if not recommendations:
            return ""
        titles = [
            f"{item.get('title', 'Unknown title')} by {item.get('author', 'Unknown author')}"
            for item in recommendations[:3]
        ]
        return "Most recent recommended books: " + "; ".join(titles) + "."

    def _format_recent_messages(self, messages: list[dict[str, str]], limit: int = 4) -> list[str]:
        """
        Convert recent thread messages into short role-prefixed strings for prompts.
        Args:
            messages: Raw thread messages.
            limit: Maximum number of recent messages to keep.
        Returns:
            A list of compact strings.
        """
        formatted: list[str] = []
        for message in messages[-limit:]:
            role = str(message.get("role", "unknown")).strip() or "unknown"
            content = summarize_text(str(message.get("content", "")).strip(), limit=160)
            if not content:
                continue
            formatted.append(f"{role}: {content}")
        return formatted
