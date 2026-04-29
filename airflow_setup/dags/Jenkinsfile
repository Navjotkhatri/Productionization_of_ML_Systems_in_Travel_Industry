pipeline {
    agent any

    stages {

        stage('Build Docker') {
            steps {
                echo 'Build Started'
            }
        }

        stage('Run Airflow') {
            steps {
                echo 'Airflow Started'
            }
        }

        stage('Trigger DAG') {
            steps {
                echo 'Flight Price DAG Triggered'
            }
        }

        stage('Check Status') {
            steps {
                echo 'Pipeline Successful'
            }
        }
    }
}