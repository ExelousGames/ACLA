pipeline{
    agent {
        node {
            label ''
            customWorkspace "/home/ec2-user/workspace/${JOB_NAME}/${BUILD_NUMBER}"
        }
    } 
    
    stages{

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
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'ACLA-server', 
                            transfers: [
                                sshTransfer(
                                    execCommand: 
                                        '''
                                        sudo docker stop $(sudo docker ps -a -q)
                                        sudo docker container prune
                                        ''', 
                                    execTimeout: 120000, 
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

        stage('deploy artifacts to server'){
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
                                        sudo docker-compose -f docker-compose.prod.yaml down
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


        stage('start docker') { 
            steps {
                 echo 'Test --deliver'
            }
        }
    }
}