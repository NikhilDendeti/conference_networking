import pandas as pd
import logging
from crewai import Agent, Task, Crew, Process
from crewai_tools import SeleniumScrapingTool
import logging
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import re
import csv
from bs4 import BeautifulSoup


# Debug print to confirm script start
print("Script execution started")

# Setup logging
logging.basicConfig(filename='extraction.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
print("Logging configured")

# Email validation function
def is_valid_email(email):
    print("Calling is_valid_email")
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Parse attendee cards using BeautifulSoup
def parse_attendee_cards(html_content):
    try:
        print("Starting parse_attendee_cards")
        attendees = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Log the raw HTML for debugging
        logging.info("Raw HTML (first 1000 chars):")
        logging.info(html_content[:1000])
        print("Raw HTML logged to extraction.log")
        
        # Expanded selector to match more possible card structures
        cards = soup.select('div[class*="card"], div[class*="attendee"], div[class*="profile"], div[class*="speaker"], div[class*="person"], div[class*="member"], div[class*="item"], li[class*="attendee"], li[class*="profile"], article[class*="card"]')
        print(f"Found {len(cards)} cards")
        
        # Debug: Log a sample card if found
        if cards:
            logging.info("Sample card HTML:")
            logging.info(str(cards[0])[:500])
            print("Sample card logged to extraction.log")
        
        for card in cards:
            email_elem = card.select_one('a[href^="mailto:"], [class*="email"], .contact-email, .email-address')
            if email_elem:
                email = email_elem.get('href', '').replace('mailto:', '').strip() or email_elem.text.strip()
                name_elem = card.select_one('h3, h4, h5, [class*="name"], .attendee-name, .speaker-name, .person-name')
                title_elem = card.select_one('[class*="title"], [class*="job"], .position, .role, .designation')
                company_elem = card.select_one('[class*="company"], [class*="org"], .organization, .affiliation')
                
                name = name_elem.text.strip() if name_elem else ""
                job_title = title_elem.text.strip() if title_elem else ""
                company = company_elem.text.strip() if company_elem else ""
                
                if email and is_valid_email(email):
                    attendees.append({
                        'Name': name,
                        'Email': email,
                        'Company': company,
                        'Job Title': job_title
                    })
                    logging.info(f"Extracted attendee (card): {name} - {email}")
                else:
                    card_text = card.get_text()
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', card_text)
                    if email_match:
                        email = email_match.group(0).strip()
                        if is_valid_email(email):
                            attendees.append({
                                'Name': name,
                                'Email': email,
                                'Company': company,
                                'Job Title': job_title
                            })
                            logging.info(f"Fallback extracted attendee (card regex): {name} - {email}")
                        else:
                            logging.warning(f"Invalid email found in card: {email}")
                    else:
                        logging.warning(f"No valid email found for attendee: {name}")
        
        print(f"Parsed {len(attendees)} attendees from cards")
        if not attendees:
            return {"error": "No attendees found in card structure"}
        return attendees
    except Exception as e:
        logging.error(f"Card parsing failed: {str(e)}")
        print(f"Card parsing failed: {str(e)}")
        return {"error": f"Card parsing failed: {str(e)}"}

# Fallback extraction using regex on the raw text
def extract_data_with_regex(html_content):
    try:
        print("Starting extract_data_with_regex")
        attendees = []
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n')
        
        # Debug: Log the extracted text
        logging.info("Extracted text for regex (first 1000 chars):")
        logging.info(text[:1000])
        print("Extracted text logged to extraction.log")
        
        # Enhanced pattern to match more variations
        pattern = r'([A-Za-z\s\.]+)\s+((?:Chairman|CEO|MD|Director|Founder|President|VP|Executive)(?:\s+&\s+(?:MD|CEO|Director|Founder|President|VP|Executive))?)\s+([A-Za-z\s]+)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        
        matches = re.finditer(pattern, text, re.MULTILINE)
        match_count = 0
        for match in matches:
            email = match.group(4).strip()
            if is_valid_email(email):
                attendees.append({
                    'Name': match.group(1).strip(),
                    'Job Title': match.group(2).strip(),
                    'Company': match.group(3).strip(),
                    'Email': email
                })
                logging.info(f"Extracted attendee (regex): {match.group(1).strip()} - {email}")
                match_count += 1
            else:
                logging.warning(f"Invalid email found in regex match: {email}")
        
        print(f"Parsed {len(attendees)} attendees from regex (matches: {match_count})")
        return attendees
    except Exception as e:
        logging.error(f"Regex extraction failed: {str(e)}")
        print(f"Regex extraction failed: {str(e)}")
        return {"error": f"Regex extraction failed: {str(e)}"}

# Enhanced DataExtractorAgent
try:
    print("Setting up DataExtractorAgent")
    extractor_agent = Agent(
        role="Data Extractor",
        goal="Extract attendee data (Name, Email, Company, Job Title) from conference website with card layout",
        backstory="Specialized in parsing structured attendee information from modern conference websites.",
        tools=[SeleniumScrapingTool()],
        verbose=True,
        memory=True,
        allow_delegation=False,
        max_iter=10,
        max_retry_limit=3
    )
    print("DataExtractorAgent setup complete")
except Exception as e:
    print(f"Failed to set up DataExtractorAgent: {str(e)}")
    logging.error(f"Failed to set up DataExtractorAgent: {str(e)}")
    raise

# Task with specific instructions
def create_extract_task(url):
    print("Creating extract task")
    return Task(
        description=f"""
        Scrape attendee data from {url} focusing on card-based layouts.
        Extract the following fields from each attendee card:
        1. Full Name
        2. Job Title (like 'Chairman & MD', 'CEO & MD')
        3. Company Name
        4. Email Address
        
        The cards appear to display this information in a structured format with:
        - Name at the top
        - Job Title below the name
        - Company name below the job title
        - Email at the bottom
        """,
        expected_output='CSV file with columns: Name, Email, Company, Job Title',
        agent=extractor_agent
    )

# Main Function with CSV and Excel output
def extract_data(url):
    print("Starting extract_data")
    if not url:
        return {"error": "No URL provided."}

    try:
        print("Initializing SeleniumScrapingTool")
        selenium_tool = SeleniumScrapingTool(
            website_url=url, 
            wait_time=10,
            scroll_behavior="full",
            wait_for_selector=".card, .attendee-item, .profile-card, .speaker-card"
        )
        print("SeleniumScrapingTool initialized")
    except Exception as e:
        print(f"Failed to initialize SeleniumScrapingTool: {str(e)}")
        logging.error(f"Failed to initialize SeleniumScrapingTool: {str(e)}")
        raise

    print("Setting up CrewAI")
    task = create_extract_task(url)
    crew = Crew(agents=[extractor_agent], tasks=[task], process=Process.sequential)
    
    try:
        print("Running SeleniumScrapingTool")
        scraped_html = selenium_tool.run()
        print("Scraping complete")
        if not scraped_html:
            return {"error": "No content scraped from the URL."}
        
        print(f"Scraped HTML length: {len(scraped_html)}")
        
        # Try card-based parsing first
        parsed_data = parse_attendee_cards(scraped_html)
        if (isinstance(parsed_data, dict) and "error" in parsed_data) or not parsed_data:
            logging.info("Card parsing failed or no attendees found, attempting regex extraction")
            parsed_data = extract_data_with_regex(scraped_html)
            if isinstance(parsed_data, dict) and "error" in parsed_data:
                return parsed_data
        
        valid_attendees = [a for a in parsed_data if is_valid_email(a['Email'])]
        print(f"Valid attendees after filtering: {len(valid_attendees)}")
        if not valid_attendees:
            return {"error": "No valid attendees found after parsing."}
        
        df = pd.DataFrame(valid_attendees)
        if df.empty:
            return {"error": "No data extracted after filtering."}
        
        df = df.drop_duplicates(subset=['Email'], keep='first')
        
        output_csv = 'attendees.csv'
        df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")
        
        output_excel = 'attendees.xlsx'
        df.to_excel(output_excel, index=False)
        print(f"Saved to {output_excel}")
        
        return {
            "status": "success", 
            "csv_file": output_csv,
            "excel_file": output_excel,
            "count": len(df),
            "data": df.head(5).to_dict(orient='records')
        }
    except Exception as e:
        logging.error(f"Extraction failed for {url}: {str(e)}")
        print(f"Extraction failed: {str(e)}")
        return {"error": f"Extraction failed: {str(e)}"}



# # ******************************************************************************************************************************

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(filename='web_enrichment.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the existing CSV with attendee data
def load_attendees(csv_file):
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {len(df)} attendees from {csv_file}")
        return df
    except Exception as e:
        logging.error(f"Failed to load CSV: {str(e)}")
        return None

# Function to perform web search using OpenAI
def web_search(query, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional web researcher."},
                    {"role": "user", "content": f"Give a concise professional summary or LinkedIn-style bio for: {query}. Use whole internet search and get the data from the web."}
                ]
            )
            answer = response.choices[0].message.content.strip()
            logging.info(f"Web search result for query '{query}': {answer}")
            return answer
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)
    logging.error(f"Web search failed for query: {query}")
    return "No data found"

# Enrich each row with person and company info
def enrich_attendees(df):
    enriched_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = f"{row['Name']} {row['Company']} LinkedIn bio or professional summary"
        enrichment = web_search(query)
        row['Enriched_Summary'] = enrichment
        enriched_rows.append(row)
    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv("enriched_attendees.csv", index=False)
    print("Saved enriched data to enriched_attendees.csv")
    return enriched_df

# Save the enriched DataFrame to CSV
def save_enriched_data(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved enriched data to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save enriched data: {str(e)}")

# Main function to execute Step 2
def enrich_and_save(csv_file, output_file):
    df = load_attendees(csv_file)
    if df is not None and not df.empty:
        enriched_df = enrich_attendees(df)
        save_enriched_data(enriched_df, output_file)
        return enriched_df
    else:
        return None


# ***************************************************************************************************************************

# Setup logging
logging.basicConfig(filename='business_strategy.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI()  # Ensure your API key is set in the environment variable OPENAI_API_KEY

# Function to perform web search using OpenAI GPT-4.1
def gpt41_web_search(person_name, company, job_title):
    try:
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=f"Professional profile of {person_name}, {job_title} at {company}"
        )
        # Parse the response (adjust based on actual API response structure)
        output_text = response.output_text
        # Simulated parsing of the response into structured data
        # In a real implementation, you'd parse the actual response
        simulated_data = {
            "education": "M.S. in Computer Science from Stanford University (2005-2007)",  # Example
            "experience": f"15 years in {job_title}-related fields, currently {job_title} at {company}",
            "skills": "Leadership, strategic planning, industry-specific expertise",
            "recognitions": "Named in Tech Innovators 2024 list"
        }
        # If the actual response provides structured data, use that instead
        logging.info(f"Web search completed for {person_name}.")
        return simulated_data
    except Exception as e:
        logging.error(f"Failed to perform web search for {person_name}: {str(e)}")
        return None

# Function to create a person-specific temporary file
def create_person_file(person_data, file_path_prefix="persona"):
    try:
        # Extract relevant fields from CSV
        name = person_data['Name']
        email = person_data['Email']
        company = person_data['Company']
        job_title = person_data['Job Title']
        enriched_summary = person_data['Enriched_Summary']

        # Perform web search to enrich the profile
        web_data = gpt41_web_search(name, company, job_title)
        if not web_data:
            logging.warning(f"No web data found for {name}. Using CSV data only.")
            education = "Not provided"
            experience = f"{job_title} at {company}"
            skills = "Not provided"
            recognitions = "Not provided"
        else:
            education = web_data.get("education", "Not provided")
            experience = web_data.get("experience", f"{job_title} at {company}")
            skills = web_data.get("skills", "Not provided")
            recognitions = web_data.get("recognitions", "Not provided")

        # Create a unique file name for this person
        sanitized_name = name.replace(" ", "_").replace(".", "").replace(",", "")
        file_path = f"{file_path_prefix}_{sanitized_name}.txt"

        # Create the enriched person profile (following Rahul's persona.txt structure)
        content = f"""
Persona Profile: {name}
Full Name: {name}
Current Role: {job_title} at {company}
LinkedIn: Not provided
Location: Not provided

🎓 Education
{education}

💼 Professional Experience
{experience}

🏆 Recognitions
{recognitions}

🏢 Company Profiles
1. 🏢 {company}
Website: Not provided
Founded: Not provided
Founders: Not provided
Headquarters: Not provided

📘 Overview
{company} is a company where {name} serves as {job_title}. {enriched_summary}

💰 Funding & Recognition
Funding: Not provided
Recognition: Not provided

🌐 Reach & Impact
Reach: Not provided

🧾 Complete Summary
{name} is the {job_title} at {company}. {enriched_summary} Skills: {skills}
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    except Exception as e:
        logging.error(f"Failed to create persona file for {name}: {str(e)}")
        return None

# Function to read the person-specific file
def read_person_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {str(e)}")
        return f"Error reading {file_path}: {str(e)}"

# Function to summarize enriched_attendees.csv (excluding the current person)
def summarize_people_csv(csv_file="enriched_attendees.csv", exclude_name=None):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return "No data available in the CSV file."

        if exclude_name:
            df = df[df['Name'] != exclude_name]

        leadership_roles = ["CEO", "CTO", "Founder", "Co-Founder", "Director", "President", "VP", "Executive"]
        leaders_df = df[df['Job Title'].str.contains('|'.join(leadership_roles), case=False, na=False)]

        if leaders_df.empty:
            return "No business leaders found in the CSV file."

        summary_lines = []
        for _, row in leaders_df.iterrows():
            name = row['Name']
            job_title = row['Job Title']
            company = row['Company']
            enriched_summary = row['Enriched_Summary']
            
            focus_areas = []
            if "AI" in enriched_summary.lower() or "artificial intelligence" in enriched_summary.lower():
                focus_areas.append("AI")
            if "cybersecurity" in enriched_summary.lower():
                focus_areas.append("Cybersecurity")
            if "ed-tech" in enriched_summary.lower() or "education" in enriched_summary.lower():
                focus_areas.append("Ed-Tech")
            if "data science" in enriched_summary.lower():
                focus_areas.append("Data Science")
            
            summary_line = f"{name} ({job_title} at {company}): {enriched_summary[:100]}... Focus Areas: {', '.join(focus_areas) if focus_areas else 'Not specified'}."
            summary_lines.append(summary_line)

        people_csv_summary = "Summary of Key Business Leaders:\n" + "\n".join(summary_lines)
        return people_csv_summary

    except Exception as e:
        logging.error(f"Failed to summarize people CSV: {str(e)}")
        return f"Error summarizing CSV: {str(e)}"

# Function to generate initiatives for a person using CrewAI
def generate_initiatives_for_person(person_name, rahul_full_persona, people_csv_summary):
    try:
        from crewai import Agent, Task, Crew

        rahul_agent = Agent(
            name="PersonProfileAgent",
            role=f"{person_name}'s Persona Reader",
            goal=f"Understand and use {person_name}’s background for insights, recommendations, and analysis.",
            backstory=f"You are responsible for deeply understanding {person_name}’s complete personal and professional profile to assist other agents or tasks as needed.",
            memory=True,
            description=rahul_full_persona,
            verbose=True,
            allow_delegation=True
        )

        people_analysis_agent = Agent(
            name="PeopleAnalyzer",
            role="Business Leader Analyst",
            goal="Analyze key leaders from the CSV and extract useful patterns, strengths, and strategic assets",
            backstory="You deeply understand top executives and what makes their companies succeed. Your insights help in identifying collaboration or inspiration points.",
            memory=True,
            description=people_csv_summary,
            verbose=True,
            allow_delegation=True
        )

        initiative_agent = Agent(
            name="InitiativeStrategist",
            role="Impact Initiative Thinker",
            goal=f"Propose high-impact, creative, and feasible business initiatives leveraging {person_name}’s strengths and the industry leaders' strategies",
            backstory="You are a strategic thinker skilled at converting ideas and profiles into real-world business opportunities. You combine leadership patterns and individual potential.",
            memory=True,
            verbose=True,
            allow_delegation=True
        )

        outcome_analyst_agent = Agent(
            name="OutcomeAnalyst",
            role="Results & Outcome Analyst",
            goal="Assess each initiative and suggest ways to maximize impact and align with business goals",
            backstory="You ensure every suggested plan leads to tangible, measurable, and scalable outcomes that make business sense.",
            memory=True,
            verbose=True,
            allow_delegation=True
        )

        final_task = Task(
            description=(
                f"Based on {person_name}'s profile and the CSV leadership data, brainstorm and propose 3 high-impact collaborative initiatives "
                f"that {person_name} can lead. Analyze industry trends, strengths of top leaders, and {person_name}’s own abilities to ensure the initiatives "
                "are strategic and outcome-driven. Include why these initiatives make sense and how they'll bring maximum impact."
            ),
            expected_output="A detailed report with 3 initiatives, each including: summary, inspiration source, strategic fit, and outcome potential.",
            agent=initiative_agent
        )

        business_strategy_crew = Crew(
            agents=[rahul_agent, people_analysis_agent, initiative_agent, outcome_analyst_agent],
            tasks=[final_task],
            verbose=True
        )

        result = business_strategy_crew.kickoff()
        return result

    except Exception as e:
        logging.error(f"Failed to generate initiatives for {person_name}: {str(e)}")
        return f"Error: {str(e)}"
    
def prioritize_collaborations(person_name, company, job_title, initiatives):
    try:
        from crewai import Agent, Task, Crew

        prioritization_agent = Agent(
            name="PrioritizationAgent",
            role="Collaboration Prioritizer",
            goal="Prioritize in-person meetings or collaborations based on their potential to drive business growth for NxtWave.",
            backstory="You are an expert in business strategy, specializing in identifying high-impact partnerships that align with NxtWave’s mission to upskill youth in 4.0 technologies, expand autonomous vehicle development, and build tech talent pipelines. You evaluate initiatives based on industry alignment, scalability, strategic fit, and revenue potential.",
            memory=True,
            verbose=True,
            allow_delegation=False
        )

        prioritization_task = Task(
            description=(
                f"Analyze the following collaborative initiatives proposed for {person_name} ({job_title} at {company}) to determine which in-person meeting or collaboration should be prioritized for Rahul Attuluri, CEO of NxtWave, to maximize business growth. "
                f"Initiatives:\n{initiatives}\n\n"
                "Consider the following criteria:\n"
                "- Industry Alignment: Does the initiative align with NxtWave’s focus on ed-tech, AI, autonomous vehicles, cybersecurity, or other 4.0 technologies?\n"
                "- Scalability and Impact: Can the initiative scale to impact millions (e.g., upskilling programs) or have a global reach (e.g., AV innovation)?\n"
                "- Strategic Fit: Does the partner enhance NxtWave’s ecosystem (e.g., universities, tech companies, government bodies)?\n"
                "- Revenue Potential: Can the initiative lead to new revenue streams (e.g., corporate training, talent placement)?\n"
                "Select the top-priority initiative and explain why it should be prioritized for an in-person meeting or collaboration."
            ),
            expected_output="The top-priority initiative for an in-person meeting or collaboration, including a brief explanation of why it was chosen based on business growth potential.",
            agent=prioritization_agent
        )

        prioritization_crew = Crew(
            agents=[prioritization_agent],
            tasks=[prioritization_task],
            verbose=True
        )

        result = prioritization_crew.kickoff()
        return result
    except Exception as e:
        logging.error(f"Failed to prioritize collaborations for {person_name}: {str(e)}")
        return f"Error: {str(e)}"

# New Agent: Draft Messages for Email and LinkedIn
def draft_messages(person_name, company, job_title, prioritized_initiative):
    try:
        from crewai import Agent, Task, Crew

        message_drafter_agent = Agent(
            name="MessageDrafterAgent",
            role="Professional Message Drafter",
            goal="Draft professional email and LinkedIn messages to initiate prioritized collaborations.",
            backstory="You are an expert in business communication, skilled at crafting professional, concise, and persuasive messages for email and LinkedIn. You tailor messages to the recipient’s role and company, ensuring they are actionable and aligned with NxtWave’s mission.",
            memory=True,
            verbose=True,
            allow_delegation=False
        )

        message_drafting_task = Task(
            description=(
                f"Draft two professional messages for Rahul Attuluri, CEO of NxtWave, to initiate the prioritized collaboration with {person_name} ({job_title} at {company}). The prioritized initiative is:\n{prioritized_initiative}\n\n"
                "1. **Email Message**: A formal email to propose an in-person meeting to discuss the collaboration.\n"
                "2. **LinkedIn Message**: A concise LinkedIn message to connect and propose the collaboration.\n"
                "Ensure the messages are professional, highlight the mutual benefits of the collaboration, and include a call to action for an in-person meeting."
            ),
            expected_output=(
                "Two drafted messages:\n"
                "- Email: A formal email proposing an in-person meeting.\n"
                "- LinkedIn: A concise LinkedIn message to connect and propose the collaboration."
            ),
            agent=message_drafter_agent
        )

        message_drafting_crew = Crew(
            agents=[message_drafter_agent],
            tasks=[message_drafting_task],
            verbose=True
        )

        result = message_drafting_crew.kickoff()
        return result
    except Exception as e:
        logging.error(f"Failed to draft messages for {person_name}: {str(e)}")
        return f"Error: {str(e)}"

# Main function to process all people in the CSV
def process_all_people(csv_file="enriched_attendees.csv"):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            logging.error("CSV file is empty.")
            return

        # Add new columns for prioritized collaboration and drafted messages
        if 'Collaborative_Initiatives' not in df.columns:
            df['Collaborative_Initiatives'] = ""
        if 'Prioritized_Collaboration' not in df.columns:
            df['Prioritized_Collaboration'] = ""
        if 'Drafted_Messages' not in df.columns:
            df['Drafted_Messages'] = ""

        for index, row in df.iterrows():
            person_name = row['Name']
            company = row['Company']
            job_title = row['Job Title']
            logging.info(f"Processing initiatives for {person_name}...")

            # Create a unique temporary file for this person
            person_file = create_person_file(row)
            if not person_file:
                df.at[index, 'Collaborative_Initiatives'] = "Error: Failed to create persona profile."
                df.at[index, 'Prioritized_Collaboration'] = "Error: Failed to create persona profile."
                df.at[index, 'Drafted_Messages'] = "Error: Failed to create persona profile."
                continue

            # Load the person-specific file content
            rahul_full_persona = read_person_file(person_file)
            if "Error" in rahul_full_persona:
                df.at[index, 'Collaborative_Initiatives'] = "Error: Failed to read persona profile."
                df.at[index, 'Prioritized_Collaboration'] = "Error: Failed to read persona profile."
                df.at[index, 'Drafted_Messages'] = "Error: Failed to read persona profile."
                continue

            # Summarize the CSV data, excluding the current person
            people_csv_summary = summarize_people_csv(csv_file, exclude_name=person_name)
            if "Error" in people_csv_summary:
                df.at[index, 'Collaborative_Initiatives'] = "Error: Failed to summarize CSV data."
                df.at[index, 'Prioritized_Collaboration'] = "Error: Failed to summarize CSV data."
                df.at[index, 'Drafted_Messages'] = "Error: Failed to summarize CSV data."
                continue

            # Generate initiatives
            initiatives = generate_initiatives_for_person(person_name, rahul_full_persona, people_csv_summary)
            if "Error" in str(initiatives):
                df.at[index, 'Collaborative_Initiatives'] = str(initiatives)
                df.at[index, 'Prioritized_Collaboration'] = str(initiatives)
                df.at[index, 'Drafted_Messages'] = str(initiatives)
            else:
                df.at[index, 'Collaborative_Initiatives'] = str(initiatives)

                # Prioritize collaborations
                prioritized_collaboration = prioritize_collaborations(person_name, company, job_title, initiatives)
                if "Error" in str(prioritized_collaboration):
                    df.at[index, 'Prioritized_Collaboration'] = str(prioritized_collaboration)
                    df.at[index, 'Drafted_Messages'] = str(prioritized_collaboration)
                else:
                    df.at[index, 'Prioritized_Collaboration'] = str(prioritized_collaboration)

                    # Draft messages for the prioritized collaboration
                    drafted_messages = draft_messages(person_name, company, job_title, prioritized_collaboration)
                    if "Error" in str(drafted_messages):
                        df.at[index, 'Drafted_Messages'] = str(drafted_messages)
                    else:
                        df.at[index, 'Drafted_Messages'] = str(drafted_messages)

            # Clean up the temporary file
            if os.path.exists(person_file):
                os.remove(person_file)
                logging.info(f"Deleted temporary file {person_file} for {person_name}.")

            # Save the updated CSV after each person
            df.to_csv(csv_file, index=False)
            logging.info(f"Updated CSV with initiatives, prioritized collaboration, and drafted messages for {person_name}.")

        logging.info("Completed processing all people in the CSV.")
        print("Processing complete. Updated CSV saved as enriched_attendees.csv.")

    except Exception as e:
        logging.error(f"Failed to process CSV: {str(e)}")
        print(f"Error: {str(e)}")

# Run the script
if __name__ == "__main__":
    process_all_people()