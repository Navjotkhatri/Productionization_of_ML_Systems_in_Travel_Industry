pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                url: 'https://github.com/Navjotkhatri/Productionization_of_ML_Systems_in_Travel_Industry.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train Model with MLflow') {
            steps {
                sh 'python mlflow_train.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t flight-price-app .'
            }
        }

        stage('Run Container') {
            steps {
                sh 'docker run -d -p 8501:8501 flight-price-app'
            }
        }

        stage('Success') {
            steps {
                echo 'Pipeline completed successfully'
            }
        }
    }
}