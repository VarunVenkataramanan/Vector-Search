This folder has 4 contents - Readme, Dockerfile, DockerCompose and all the necessary files 					to run.
Change directory to "./tagalys"
Run "docker compose up --build"
You should have 3 links that appear
Open the link which starts with http://127.0.0.0:8888....
You should have Jupyter notebook running
The only file of interests are -
	"Fashion_preprocess.ipynb", which creates the text and image embeddings
	"final_embeddings.csv", which is the CSV file containing the image and text embeddings
Select the respective names to open in a new tab
All the other files are used in the github from where we took this project
In another tab, run "docker compose exec search bash", if you want to open a terminal in this container

