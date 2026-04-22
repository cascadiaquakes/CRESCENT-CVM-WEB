import aws_cdk as core
import aws_cdk.assertions as assertions

from cvmweb_app513_f4_fcd.cvmweb_app513_f4_fcd_stack import CvmwebApp513F4FcdStack

# example tests. To run these tests, uncomment this file along with the example
# resource in cvmweb_app513_f4_fcd/cvmweb_app513_f4_fcd_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = CvmwebApp513F4FcdStack(app, "cvmweb-app513-f4-fcd")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
