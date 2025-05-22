pipeline{
    agent {
        node {
            label ''
            customWorkspace "/home/ec2-user/workspace/${JOB_NAME}/${BUILD_NUMBER}"
        }
    } 
    
    stages{

        stage('get workspace directory docker'){
            steps{
                 echo "Current workspace: ${env.WORKSPACE}"
            }
        }

        stage('clean built docker image'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml down'

            }
        }

        stage('build docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env build'
            }
        }

        stage('stop server'){
            steps{
                echo 'frontend tested'
            }
        }

        stage('push artifacts to server'){
            steps{
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-server', 
                            transfers: [
                                sshTransfer(
                                    cleanRemote: false, 
                                    excludes: '', 
                                    execCommand: '', 
                                    execTimeout: 120000, 
                                    flatten: false, 
                                    makeEmptyDirs: false, 
                                    noDefaultExcludes: false, 
                                    patternSeparator: '[, ]+', 
                                    remoteDirectory: '', 
                                    remoteDirectorySDF: false, 
                                    removePrefix: '', 
                                    sourceFiles: "/home/ec2-user/workspace/${JOB_NAME}/${BUILD_NUMBER}"
                                )
                            ], 
                            usePromotionTimestamp: false, 
                            useWorkspaceInPromotion: false, 
                            verbose: false)])
            }
        }


        stage('start docker') { 
            steps {
                 echo 'Test --deliver'
            }
        }
    }
}