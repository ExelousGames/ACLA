def cleanEncoding(value):
        if isinstance(value, (str, bytes)):
            return value.encode('ascii', errors='ignore').decode('ascii').replace('\u0000', '')
        return value