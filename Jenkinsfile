pipeline{
    agent any 
    stages{
        stage('build frontend'){
            steps{
                sh 'cd acla_backend'
                sh 'npm install'
                sh 'cd ..'
            }
        }

        stage('build backend'){
            steps{
                sh 'cd acla_front'
                sh 'npm install'
                sh 'cd ..'
            }
        }

        stage('build desktop'){
            steps{
                sh 'cd desktop_application/build'
                sh 'pip install -r requirements.txt'
                sh 'cd ..'
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