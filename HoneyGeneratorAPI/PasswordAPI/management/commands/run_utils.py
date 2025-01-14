# Run this file using "python manage.py run_utils" to allow utils.py for this script to run within Django's environment and leverage all of its configurations.
from django.core.management.base import BaseCommand
from PasswordAPI.utils import *  # Import your utility function

class Command(BaseCommand):
    help = 'Run the utils.py script'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting utils.py logic...")
        run_utils()  # Function to call to test utils.py
        self.stdout.write("Finished running utils.py logic.")