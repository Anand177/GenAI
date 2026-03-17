from agents import Runner, trace, gen_trace_id
from EmailAgent import email_agent
from PlannerAgent import planner_agent, WebSearchItem, WebSearchPlan
from SearchAgent import search_agent
from WriterAgent import writer_agent, ReportData

import asyncio

class Researchmanager:
    
    async def plan_searches(self, query: str) -> WebSearchPlan:
        """Plan searches to perform for the query"""
        print("Planning searches..")
        result = await Runner.run(planner_agent, f"Query : {query}")

        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)
    

    async def perform_search(self, search_plan: WebSearchPlan) -> list[str]:
        """Perform searches to perform for the query"""
        print("Searching...")
        num_completed = 0
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """ Perform a search for the query """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Write the report for the query """
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(
            writer_agent,
            input,
        )

        print("Finished writing report")
        return result.final_output_as(ReportData)
    
    async def send_email(self, report: ReportData) -> None:
        print("Writing email...")
        result = await Runner.run(
            email_agent,
            report.markdown_report,
        )
        print("Email sent")
        return report
