# Conference Networking Application

## Project Overview

The **Conference Networking Application** aims to enhance networking at conferences by automating the extraction, enrichment, and analysis of attendee data. The system helps identify high-value connections, generates personalized outreach messages, and prioritizes in-person meetings. The backend leverages CrewAI for data processing and analysis, while the frontend is designed to be built with Streamlit.

## Features

- **Attendee Data Extraction:** Automatically pulls attendee information from the conference website.
- **Data Enrichment:** Enhances extracted data with personal and company information sourced from the internet.
- **Summarization:** Generates summarized data in Excel for easy access and analysis.
- **Business Relationship Identification:** Analyzes data to identify potential high-value business relations.
- **Outreach Message Generation:** Drafts personalized outreach messages for LinkedIn and email.

## Architecture

The backend of this application is built in Python using the following technologies:

- **CrewAI:** For intelligent data processing and analysis.
- **Streamlit:** For future frontend development.
- **Excel Summarization:** For generating summarized attendee information.
- **API Integration:** Integrates external APIs for enriching attendee data and generating personalized outreach messages.

This is the **base version** of the backend, with a focus on extracting, enriching, and summarizing data. Future development will focus on building an interactive frontend using **Streamlit**.

## Technologies Used

- Python 3.x
- CrewAI
- Pandas (for data processing)
- OpenAI/Groq API (for AI-driven content generation)
- Excel (for summarization)

## Installation

### Prerequisites

- Python 3.8+
- A valid API key for CrewAI and OpenAI/Groq (if applicable)
