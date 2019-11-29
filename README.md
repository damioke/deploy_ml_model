## google cloud account setup:
- create a gmail account
- go to: console.cloud.google.com and sign in with your newly created gmail account
- activate the free trial by. you will need a credit card for this.
- welcome to the cloud, you are ready to rock and roll with a free $300 or 12 months free usage, whichever comes first.

## configuration and deployment:
- click on the shell icon on the right top right to activate a vm
- launch the editor to get view of your file system
- to clone the source code run this command: git clone https://github.com/damioke/deploy_ml_model.git
- switch to the application folder with this command: cd deploy_ml_model/APP_HOME/
- give permission to the deploy script with this command: chmod a+x deploy.sh
- change PROJECT_ID="YOUR_PROJECT_ID"
- enable the cloud Build and cloud Run APIs from this link: https://cloud.google.com/run/docs/quickstarts/build-and-deploy
- deploy the application using this command: ./deploy.sh
- use postman to test your application
- Done