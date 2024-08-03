import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, ScrapeWebsiteTool
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import PrivateAttr, Field
import re

class DataFrameUpdateTool(BaseTool):
    name: str = "DataFrameUpdateTool"
    description: str = "Tool for updating, retrieving, and managing player data in a pandas DataFrame."
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, action: str, **kwargs: Any) -> str:
        if action == "get_players":
            return self.get_players()
        elif action == "update_player":
            return self.update_player(**kwargs)
        elif action == "get_dataframe":
            return self.get_dataframe()
        else:
            return f"Unknown action: {action}. Available actions are: get_players, update_player, get_dataframe"

    def get_players(self) -> List[str]:
        return self._df['name'].tolist()

    def update_player(self, **kwargs) -> str:
        player_name = kwargs.get('name')
        if not player_name:
            return "No player name provided."
        
        if player_name not in self._df['name'].values:
            return f"Player {player_name} not found in the DataFrame."
        
        row_index = self._df.index[self._df['name'] == player_name].tolist()[0]
        for column, value in kwargs.items():
            if column in self._df.columns:
                self._df.at[row_index, column] = value
        
        return f"Updated data for player {player_name}"

    def get_dataframe(self) -> str:
        return self._df.to_dict(orient='records')

def create_nhl_data_crew(excel_file: str, capwages_url: str):
    df = pd.read_excel(excel_file)
    
    scrape_tool = ScrapeWebsiteTool(website_url=capwages_url)
    df_update_tool = DataFrameUpdateTool(df)

    research_agent = Agent(
        role="Senior NHL Data Research Specialist",
        goal="Update Seattle Kraken player data using Capwages, ensuring all players in the original dataset are accounted for",
        backstory="You are a seasoned sports researcher with extensive knowledge of NHL statistics and player contracts, specializing in the Seattle Kraken.",
        verbose=True,
        allow_delegation=False,
        tools=[scrape_tool, df_update_tool]
    )

    entry_agent = Agent(
        role="Senior NHL Data Entry Specialist",
        goal="Update the DataFrame with new Seattle Kraken player information, using only existing fields and ensuring all players are processed",
        backstory="You have years of experience in data entry for NHL teams, with a focus on accuracy and consistency in player statistics.",
        verbose=True,
        allow_delegation=False,
        tools=[df_update_tool]
    )

    accuracy_agent = Agent(
        role="Senior Data Accuracy Specialist",
        goal="Ensure the integrity and accuracy of the updated Seattle Kraken player data, verifying all players have been processed",
        backstory="With a background in statistical analysis and data validation for NHL teams, you have a keen eye for inconsistencies in hockey player data.",
        verbose=True,
        allow_delegation=True,
        tools=[df_update_tool]
    )

    research_task = Task(
        description=f"""
        1. Use the DataFrameUpdateTool with the action "get_players" to get a list of all players in the original dataset.
        2. Use the scrape_tool to extract data from the Capwages page for the Seattle Kraken: {capwages_url}
        3. For each player in the original dataset:
           a. Attempt to find matching data on the Capwages page, considering potential name format differences.
           b. If found, collect updated data from Capwages, focusing ONLY on information that matches the existing fields in our DataFrame.
           c. If not found on Capwages, note this for the player.
           d. Format the data for each player as a dictionary, including a status field indicating if data was found or not.
        4. Return a list of dictionaries, each containing the updated information (or status) for a player.
        """,
        agent=research_agent,
        expected_output="A list of dictionaries containing updated information or status for all players in the original dataset."
    )

    entry_task = Task(
        description="""
        1. You will receive a list of dictionaries with updated Seattle Kraken player data from the Research Specialist.
        2. For each player dictionary:
           a. Use the DataFrameUpdateTool with the action "update_player" to update the player's information in the DataFrame.
           Example: df_update_tool._run(action="update_player", name="Player Name", cap_hit="$1,000,000", ...)
           b. Log any players that couldn't be updated or any issues encountered.
        3. Ensure that all players from the original dataset are accounted for in your updates.
        4. Return a summary of the updates made and any issues encountered, including players not found on Capwages.
        """,
        agent=entry_agent,
        expected_output="A summary report of the DataFrame updates, including successful updates, issues encountered, and players not found on Capwages."
    )

    accuracy_task = Task(
        description=f"""
        1. Use the DataFrameUpdateTool with the action "get_dataframe" to review the updated DataFrame for accuracy and completeness.
        2. Cross-reference the updated data with the Capwages page ({capwages_url}) to ensure no errors occurred during research or entry.
        3. Check for any inconsistencies, outliers, or suspicious data points that may indicate an error.
        4. Verify that all players from the original dataset have been accounted for, even if their data wasn't updated.
        5. If any discrepancies are found, use the DataFrameUpdateTool with the action "update_player" to correct them in the DataFrame.
        6. If major issues are found that you can't resolve, delegate the task back to the Research or Entry agent as appropriate.
        7. Prepare a detailed report on the accuracy of the data for the Seattle Kraken, including:
           - Total number of players in the original dataset
           - Number of players updated with new data
           - Number of players not found on Capwages
           - Types of updates made
           - Any issues or discrepancies found and how they were resolved
           - Overall assessment of data quality
        """,
        agent=accuracy_agent,
        expected_output="A comprehensive accuracy report detailing the verification process, corrections made, and an overall assessment of the data quality for all players."
    )

    nhl_data_crew = Crew(
        agents=[research_agent, entry_agent, accuracy_agent],
        tasks=[research_task, entry_task, accuracy_task],
        verbose=2,
        process=Process.sequential
    )

    return nhl_data_crew, df_update_tool

def main():
    excel_file = "Players_exported_1.xlsx"
    capwages_url = "https://capwages.com/teams/seattle_kraken"
    
    crew, df_tool = create_nhl_data_crew(excel_file, capwages_url)
    result = crew.kickoff()

    # Save the updated DataFrame to a new Excel file
    updated_df = pd.DataFrame(df_tool._run(action="get_dataframe"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file_name = f"updated_seattle_kraken_players_{timestamp}.xlsx"
    updated_df.to_excel(new_file_name, index=False)
    print(f"Updated data saved to {new_file_name}")

    # Print the full result for debugging
    print("Full result from crew:")
    print(result)

if __name__ == "__main__":
    main()