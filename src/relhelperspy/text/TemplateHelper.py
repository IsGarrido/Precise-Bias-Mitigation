from relhelperspy.primitives.annotations import log_time

class TemplateHelper:

    @log_time
    @staticmethod
    def generate(templates, values, include_header = False):

        print(str(len(templates)) + " templates")
        print(str(len(values)) + " values")

        lines = []
        for template in templates:
            sentences = []
            for val in values:
                try:
                    sentences.append(template.format(value=val)) 
                except:
                    print("Error in template, value: " + str(template) + ", " + str(val))

            line = str.join("\t", sentences)
            lines.append(line)
        
        if include_header:
            header = str.join("\t", values)
            lines.insert(0, header)

        
        return lines