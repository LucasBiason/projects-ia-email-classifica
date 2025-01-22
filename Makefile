
runapp:
	docker-compose up --build -d;

runapp-dev:
	docker-compose down;
	docker-compose up --build; 