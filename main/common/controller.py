
# The controller functions as the connection between the user and the application
# User input is passed from the view to the model by the controller
class Controller():
    def __init__(self, model):
        self.model = model

    # temp function to calculate the blob diameters
    # in the future tracking will be added
    def process_data(self, directory):
        directory = directory + "/*tif"
        # inladen van sample set
        self.model.load_sampleset(directory)
        # detecteren van blobs
        self.model.detect_blobs(self.model.image_array)
        self.model.blob_tracking(self.model.image_array)
        return self.model.get_blob_diameters()
