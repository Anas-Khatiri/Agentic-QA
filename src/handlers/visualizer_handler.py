from src.agents.analyzer_agent import AnalyzerAgent
from src.agents.comparator_agent import ComparatorAgent
from src.agents.vehicle_sales_visualizer_agent import VehicleSalesVisualizerAgent


def visualize_vehicles_sold_per_year(streamlit: bool = False) -> None:
    """Extract and plot the number of Renault vehicles sold per year (from 2020)."""
    agent = VehicleSalesVisualizerAgent()
    agent.plot(streamlit=streamlit)


def visualize_stock_vs_index(streamlit: bool = False) -> None:
    """Generate a line graph comparing Renault's stock price to the CAC40 index on earnings announcement dates since 2020."""
    agent = ComparatorAgent()
    agent.plot(streamlit=streamlit)


def visualize_sales_vs_stock_correlation(streamlit: bool = False) -> None:
    """Plot a scatter graph showing the correlation between annual vehicle sales and Renault's average stock price on earnings days (from 2020 onwards)."""
    agent = AnalyzerAgent()
    agent.plot(streamlit=streamlit)
