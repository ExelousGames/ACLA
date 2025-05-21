pipeline{
    agent any 
    stages{
        stage('init docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env up'

            }
        }

        stage('build backend'){
            steps{
                echo 'backend tested'
            }
        }

        stage('build desktop'){
            steps{
                echo 'desktop tested'
            }
        }

        stage('test frontend'){
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