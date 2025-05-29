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

        stage('Testing nothing'){
            steps{
                echo 'Tested'
            }
        }

        stage('Package database deployment') {
            steps {
                sh '''
                    sudo rm deployment.zip || true
                    sudo rm -r deployment/ || true
                    mkdir -p deployment
                    cp -R acla_db/ deployment/
                    cp docker-compose.prod.yaml .prod.env deployment/
                    zip -r deployment.zip deployment/
                '''
                archiveArtifacts artifacts: 'deployment.zip', fingerprint: true
            }
        }

        stage('Stop and clean database'){
            steps{
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-database', 
                            transfers: [
                                sshTransfer(
                                    execCommand: 
                                        '''
                                        cd deployment
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env down
                                        yes | sudo docker container prune
                                        yes | sudo docker image prune
                                        yes | sudo docker volume prune
                                        cd ..
                                        sudo rm deployment.zip || true
                                        sudo rm -r deployment/ || true
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


        stage('Deploy to database'){
            steps{

                echo "${env.WORKSPACE}/deployment.zip file will be pushed "

                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-database', 
                            transfers: [
                                sshTransfer(
                                    cleanRemote: false, 
                                    excludes: '', 
                                    execCommand: 
                                    '''
                                        unzip deployment.zip
                                        cd deployment
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env up -d --build 'mongodb'
                                    '''
                                    , 
                                    execTimeout: 7200000, 
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


        stage('Package frontend and backend deployment') {
            steps {
                sh '''
                    sudo rm deployment.zip || true
                    sudo rm -r deployment/ || true
                    mkdir -p deployment
                    cp -R acla_backend/ acla_front/ backend_nginx/ deployment/
                    cp docker-compose.prod.yaml .prod.env deployment/
                    zip -r deployment.zip deployment/
                '''
                archiveArtifacts artifacts: 'deployment.zip', fingerprint: true
            }
        }

        stage('Stop and clean backend server'){
            steps{
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-backend', 
                            transfers: [
                                sshTransfer(
                                    execCommand: 
                                        '''
                                        cd deployment
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env down
                                        yes | sudo docker container prune
                                        yes | sudo docker image prune
                                        yes | sudo docker volume prune
                                        cd..
                                        sudo rm deployment.zip || true
                                        sudo rm -r deployment/ || true
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


        stage('Deploy to frontend and backend server'){
            steps{

                echo "${env.WORKSPACE}/deployment.zip file will be pushed "

                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-backend', 
                            transfers: [
                                sshTransfer(
                                    cleanRemote: false, 
                                    excludes: '', 
                                    execCommand: 
                                    '''
                                        unzip deployment.zip
                                        cd deployment
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env up -d --build 'frontend' 'backend_proxy' 'backend'
                                    '''
                                    , 
                                    execTimeout: 7200000, 
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