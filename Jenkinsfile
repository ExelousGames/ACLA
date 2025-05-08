pipeline{
    agent any 
    stages{
        stage('build frontend'){
            steps{
                sh 'npm install --prefix acla_backend'

            }
        }

        stage('build backend'){
            steps{
                sh 'npm install --prefix acla_front'
            }
        }

        stage('build desktop'){
            steps{
                sh 'python3 --version'
                sh 'pip3 install --target ./desktop_application/build/ -r requirements.txt'

            }
        }

        stage('test frontend'){
            steps{
                echo 'test frontend'
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
                 echo ' test Deliver'
            }
        }
    }
}