Deployment instructions for the "Ball Pathway Prediction Application" involve setting up the necessary environment, installing dependencies, and running the application.
1. Python Installation
Ensure that Python 3.10 is installed on your system. 
If not, download and install Python from the official website and follow the install instruction .
After installation open cmd in your target folder and check the python 3.10 is available. Because above 3.10 python version is required.
	
	python --version
  	
If you can’t find python 3.10 in cmd add your path in environmental variable and move to top in the list . 
Next step is clone project from git repository.
  
 	git clone https://github.com/smsivanujan/ball_pathway_prediction_application.git

After that change directory path to where our project placed in folder.

    cd ball_pathway_prediction_application\ball_pathway_prediction

2. Virtual Environment
Create a virtual environment to isolate dependencies for the application.

        pip install virtualenv
        virtual venv
        venv\Scripts\activate

3. Dependency Installation
Install the necessary Python libraries in requirements.txt using pip.
	
        D:\ball_pathway_prediction_application\ball_pathway_prediction\venv\Scripts\python.exe -m pip install -r requirements.txt
	
4. Run the Application
Run the Python script to launch the application.

        python main.py

Enjoy

THIS APPLICATION IS CREATED BY SM SIVANUJAN FOR DESSERTATION PROJECT OF ICBT JAFFNA BSC IT.
   
