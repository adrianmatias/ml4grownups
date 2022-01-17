# ml4grownups rest api



cd ml4grownups/src/restapi/app
uvicorn main:app --reload

yeah




docker build -t ml4grownups:rest-api .
docker run --rm -p 80:80 ml4grownups:rest-api

ml4grownups:rest-api