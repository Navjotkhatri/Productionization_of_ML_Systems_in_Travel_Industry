pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                url: 'https://github.com/Navjotkhatri/Productionization_of_ML_Systems_in_Travel_Industry.git'
            }
        }

        stage('Install Python') {
            steps {
                sh '''
                apt update
                apt install -y python3 python3-pip
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                pip3 install -r requirements.txt
                '''
            }
        }

        stage('Train Model with MLflow') {
            steps {
                sh '''
                python3 mlflow_train.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t flight-price-app .
                '''
            }
        }

        stage('Run Docker Container') {
            steps {
                sh '''
                docker rm -f flight-container || true
                docker run -d -p 5001:5001 --name flight-container flight-price-app
                '''
            }
        }

        stage('Success') {
            steps {
                echo 'CI/CD Pipeline Executed Successfully'
            }
        }
    }
}