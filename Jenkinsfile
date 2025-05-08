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
                sh 'python -V'
                sh 'python -m ensurepip --upgrade'
                sh 'pip install -r --target ./desktop_application/build requirements.txt'

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