pipeline{
    agent any 
    stages{
        stage('clean docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml down'

            }
        }

        stage('build docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env up'
            }
        }

        stage('stop docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml down'
            }
        }

        stage('test'){
            steps{
                echo 'frontend tested'
            }
        }

        stage('test backend'){
            steps{
                echo 'test backend'
            }
        }

        stage('test desktop'){
            steps{
                echo 'test desktop'
            }
        }


        stage('Deliver') { 
            steps {
                 echo 'Test --deliver'
            }
        }
    }
}