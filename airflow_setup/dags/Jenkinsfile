pipeline {
    agent any

    stages {

        stage('Clone Repo') {
            steps {
                git 'https://github.com/YOUR_USERNAME/YOUR_REPO.git'
            }
        }

        stage('Build Docker') {
            steps {
                sh 'docker compose build'
            }
        }

        stage('Run Airflow') {
            steps {
                sh 'docker compose up -d'
            }
        }

        stage('Trigger DAG') {
            steps {
                sh '''
                sleep 20
                curl -X POST "http://localhost:8080/api/v1/dags/flight_price_pipeline/dagRuns" \
                -H "Content-Type: application/json" \
                -u airflow:airflow \
                -d '{}'
                '''
            }
        }

        stage('Check Status') {
            steps {
                echo "Pipeline executed successfully!"
            }
        }
    }
}
