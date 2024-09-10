import builtins
import logging
import sys

class PrintLogger:
    def __init__(self, log_file='output.log', log_level=logging.INFO):
        
        self.original_print = builtins.print

        # Set up the logger
        self.logger = logging.getLogger('PrintLogger')
        self.logger.setLevel(log_level)
        
        # Create handlers
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.file_handler = logging.FileHandler(log_file)
        
        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console_handler.setFormatter(formatter)
        self.file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
        
    def overwrite_print(self):
        def custom_print(*args, **kwargs):
            message = ' '.join(map(str, args))
            self.logger.info(message)
            # self.original_print(*args, **kwargs)

        # Overwrite the built-in print function
        builtins.print = custom_print
        return self

    def set_level(self, level):
        self.logger.setLevel(level)
        self.console_handler.setLevel(level)
        self.file_handler.setLevel(level)
        
# print_logger = PrintLogger().setup_handler("path").overwrite_print()

