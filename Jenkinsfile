pipeline {

  environment {
    AIC_REGISTRY = "60.250.28.238:30032"
  }

  agent {
    kubernetes {
      defaultContainer 'ci-runner'
      yamlFile 'kubernetes-agent-pod.yaml'
    }
  }

  options {
    ansiColor('xterm')
    timeout(time: 30, unit: 'MINUTES')
  }

  stages {


    stage('Clean') {
      steps {
        container('ci-runner') {
          sh 'make clean'
        }
      }
    }

    stage('Build Process') {
      steps {
        container('ci-runner') {
          sh 'make build-process'
        }
      }
    }

    stage('Push Process IMG') {
      steps {
        withCredentials([[$class: "UsernamePasswordMultiBinding",
                           credentialsId: "aic-registry",
                           usernameVariable: "USERNAME",
                           passwordVariable: "PASSWORD"]]) {
          container('ci-runner') {
              sh '''
                docker login --username "$USERNAME" --password="$PASSWORD" ${AIC_REGISTRY}
                make push-process-img
              '''
          }
        }
      }
    }



    stage('Build Train') {
      steps {
        container('ci-runner') {
          sh 'make build-train'
        }
      }
    }

    stage('Push Train IMG') {
      steps {
        withCredentials([[$class: "UsernamePasswordMultiBinding",
                           credentialsId: "aic-registry",
                           usernameVariable: "USERNAME",
                           passwordVariable: "PASSWORD"]]) {
          container('ci-runner') {
              sh '''
                docker login --username "$USERNAME" --password="$PASSWORD" ${AIC_REGISTRY}
                make push-train-img
              '''
          }
        }
      }
    }


  }


  post {
    failure {
      updateGitlabCommitStatus name: 'ml-env-containers-build', state: 'failed'
    }
    success {
      updateGitlabCommitStatus name: 'ml-env-containers-build', state: 'success'
    }
    always {
      container('ci-runner') {
          sh '''
            make clean
          '''
      }
    }

  }

}

