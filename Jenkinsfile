pipeline{
    agent {
        node {
            label ''
            customWorkspace "/home/ec2-user/workspace/${JOB_NAME}/${BUILD_NUMBER}"
        }
    } 
    
    stages{

        stage('Clean docker'){
            steps{
                sh 'yes | sudo docker system prune -a --volumes'

            }
        }

        stage('Build docker'){
            steps{
                sh 'sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env build'
            }
        }

        stage('Stop server'){
            steps{
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-server', 
                            transfers: [
                                sshTransfer(
                                    execCommand: 
                                        '''
                                        sudo docker stop $(sudo docker ps -a -q)
                                        yes | sudo docker container prune
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

        stage('Deploy artifacts to server'){
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
                                        cd deployment
                                        unzip deployment.zip
                                        sudo docker-compose -f docker-compose.prod.yaml --env-file .prod.env up -d
                                        ''', 
                                    execTimeout: 120000, 
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