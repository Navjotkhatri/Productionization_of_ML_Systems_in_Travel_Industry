pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git 'https://github.com/Navjotkhatri/Productionization_of_ML_Systems_in_Travel_Industry.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
                bat 'pip install mlflow lightgbm xgboost'
            }
        }

        stage('Train Model with MLflow') {
            steps {
                bat 'python mlflow_train.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t travel-ml-app .'
            }
        }

        stage('Run Container') {
            steps {
                bat 'docker run -d -p 5000:5000 --name travelapp travel-ml-app'
            }
        }

        stage('Success') {
            steps {
                echo 'Pipeline completed successfully'
            }
        }
    }
}