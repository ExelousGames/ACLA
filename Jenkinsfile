pipeline{
    agent {
        node {
            label ''
            customWorkspace "/home/ec2-user/workspace/${JOB_NAME}/${BUILD_NUMBER}"
        }
    } 
    
    stages{

        stage('Clean docker files'){
            steps{
                sh 'yes | sudo docker system prune -a --volumes'

            }
        }

        stage('Build docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env build'
            }
        }

        stage('Package Deployment') {
            steps {
                sh '''
                    mkdir -p deployment
                    cp -R acla_backend/ acla_db/ acla_front/ backend_nginx/ deployment/
                    cp docker-compose.prod.yaml .prod.env deployment/
                    zip -r deployment.zip deployment/
                '''
                archiveArtifacts artifacts: 'deployment.zip', fingerprint: true
            }
        }


        stage('Stop and clean server'){
            steps{
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-server', 
                            transfers: [
                                sshTransfer(
                                    execCommand: 
                                        '''
                                        set -x
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env down
                                        yes | sudo docker container prune
                                        yes | sudo docker image prune
                                        yes | sudo docker volume prune
                                        sudo rm deployment.zip
                                        sudo rm -r deployment/
                                        ''', 
                                    execTimeout: 600000, 
                                )
                            ], 
                            usePromotionTimestamp: false, 
                            useWorkspaceInPromotion: false, 
                            verbose: false)]
                    )
            }
        }


        stage('Deploy to server'){
            steps{

                echo "${env.WORKSPACE}/deployment.zip file will be pushed "

                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-server', 
                            transfers: [
                                sshTransfer(
                                    cleanRemote: false, 
                                    excludes: '', 
                                    execCommand: 
                                    '''
                                        unzip deployment.zip
                                        cd deployment
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env up -d
                                    '''
                                    , 
                                    execTimeout: 600000, 
                                    flatten: false, 
                                    makeEmptyDirs: false, 
                                    noDefaultExcludes: false, 
                                    patternSeparator: '[, ]+', 
                                    remoteDirectory: '', 
                                    remoteDirectorySDF: false, 
                                    removePrefix: '', 
                                    sourceFiles: "deployment.zip"
                                )
                            ], 
                            usePromotionTimestamp: false, 
                            useWorkspaceInPromotion: false, 
                            verbose: false)]
                    )
            }
        }
    }
}