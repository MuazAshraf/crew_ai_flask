import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Agent, Task, Crew, Process
from flask import Flask, render_template, request
from flask_cors import CORS
app = Flask(__name__, static_folder='static')
CORS(app)
from dotenv import load_dotenv
load_dotenv('.env', override = True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
search_tool = TavilySearchResults(max_results=1)


@app.route('/', methods=['GET'])
def index():
    return render_template('crew_ai.html', result=None)


@app.route('/execute-tasks', methods=['POST'])
def execute_tasks():
    # Capture form data
    researcher_role = request.form['researcher_role']
    researcher_goal = request.form['researcher_goal']
    researcher_backstory = request.form['researcher_backstory']
    researcher_model = request.form['researcher_model']
    
    writer_role = request.form['writer_role']
    writer_goal = request.form['writer_goal']
    writer_backstory = request.form['writer_backstory']
    
    task1_description = request.form['task1_description']
    task2_description = request.form['task2_description']

    # Create Agent and Task objects dynamically
    researcher = Agent(
    role=researcher_role, 
    goal=researcher_goal, 
    backstory=researcher_backstory, 
    verbose=True, 
    allow_delegation=False,llm=ChatOpenAI(model_name=researcher_model, temperature=0.7),tools=[search_tool])
    writer = Agent(role=writer_role, goal=writer_goal, backstory=writer_backstory, verbose=True, allow_delegation=False)

    task1 = Task(description=task1_description, agent=researcher)
    task2 = Task(description=task2_description, agent=writer)

    # Instantiate your crew with a sequential process
    crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=2, process=Process.sequential)

    # Get your crew to work!
    result = crew.kickoff()

    print("######################")
    print(result)
     # Render the same page but with the result
    return result

if __name__ == '__main__':
    app.run(debug=True)