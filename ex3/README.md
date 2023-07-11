# Data-intensive Computing, exercise 3: Code offloading
This is a collection of code we wrote to solve exercise 3 of the Data-intensive Computing course at TU Wien.

TODO: explain directory structure

## AWS instructions
This is a collection of things that weren't that straightforward to figure out when using AWS.

### Get to AWS Management Console (with AWS Learner Lab Account)
1. Login to Learner Lab
2. Navigate to Dashboard
3. Click 'Start'
4. When 'light'/indicator next to link called 'AWS' becomes green, click it to get to AWS console

### Access to AWS CLI from local device with Learner Lab account
In Learner Lab Dashboard:
1. Click on 'AWS Details'
2. Click on 'Show' button next to 'AWS CLI'
3. Follow instructions (copying the shown config to the specified folder)

### Store credentials for root user of private AWS account for use with AWS CLI
1. Follow instructions for creating access key [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_root-user.html#id_root-user_manage_add-key)
2. Use `aws config` to store the created access key (+ id) in the credentials file (automatically used by CLI). Recommended settings:
   - default region name: eu-central-1 (Frankfurt, closest to Vienna)
   - default output format: json