pipeline{
    agent any 
    stages{
        stage('init docker'){
            steps{
                echo 'docker tested'

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