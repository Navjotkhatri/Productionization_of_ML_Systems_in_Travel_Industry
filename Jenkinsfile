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
                sh '''
                python3 -m pip install --break-system-packages -r requirements.txt

                python3 mlflow_train.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t flight-price-app .'
            }
        }

        stage('Run Flask Container') {
            steps {
                sh '''
                docker stop flight-app || true
                docker rm flight-app || true

                docker run -d -p 8000:8000 --name flight-app flight-price-app
                '''
            }
        }
    }
}