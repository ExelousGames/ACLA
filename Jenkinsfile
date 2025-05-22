pipeline{
    agent any 
    node {
      customWorkspace "${JENKINS_HOME}/workspace/${JOB_NAME}/${BUILD_NUMBER}"
    }
    stages{
        stage('clean docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml down'

            }
        }

        stage('build docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env build'
            }
        }

        stage('clean docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml down'
            }
        }

        stage('deploy to server'){
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