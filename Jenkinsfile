pipeline{
    agent any 
    stages{
        stage('init docker'){
            steps{
                sh 'sudo yum update -y ; sudo amazon-linux-extras install docker ; sudo service docker start'

            }
        }

        stage('build backend'){
            steps{
            }
        }

        stage('build desktop'){
            steps{
            }
        }

        stage('test frontend'){
            steps{
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