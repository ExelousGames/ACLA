def cleanEncoding(value):
        if isinstance(value, (str, bytes)):
            return value.encode('ascii', errors='ignore').decode('ascii')
        return value