pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                url: 'https://github.com/Navjotkhatri/Productionization_of_ML_Systems_in_Travel_Industry.git'
            }
        }

        stage('Train Model with MLflow') {
            steps {
                sh 'python3 -m pip install -r requirements.txt'
                sh 'python3 mlflow_train.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t flight-price-app .'
            }
        }

        stage('Run Flask Container') {
            steps {
                sh 'docker stop flight-app || true'
                sh 'docker rm flight-app || true'
                sh 'docker run -d -p 8000:8000 --name flight-app flight-price-app'
            }
        }

        stage('Success') {
            steps {
                echo 'Pipeline completed successfully'
            }
        }
    }
}