class PGNReader:
    RESULT_STRING_LIST = ["*", "1-0", "0-1", "1/2-1/2"]

    @staticmethod
    def read_from_pgn(file_name):
        with open(file_name, encoding="GB2312") as file:
            flines = file.readlines()

        lines = []
        for line in flines:
            it = line.strip()

            if len(it) == 0:
                continue

            lines.append(it)

        lines = PGNReader.__get_headers(lines)
        lines, docs = PGNReader.__get_comments(lines)
        return PGNReader.__get_steps(lines)

    @staticmethod
    def __get_headers(lines):
        index = 0
        for line in lines:

            if line[0] != "[":
                return lines[index:]

            if line[-1] != "]":
                raise Exception("Format Error on line %d" % (index + 1))

            items = line[1:-1].split("\"")

            if len(items) < 3:
                raise Exception("Format Error on line %d" % (index + 1))

            index += 1

    @staticmethod
    def __get_comments(lines):
        if lines[0][0] != "{":
            return lines, None

        docs = lines[0][1:]

        if docs[-1] == "}":
            return lines[1:], docs[:-1].strip()

        index = 1

        for line in lines[1:]:
            if line[-1] == "}":
                docs = docs + "\n" + line[:-1]
                return lines[index + 1:], docs.strip()

            docs = docs + "\n" + line
            index += 1

        raise Exception("Comments not closed")

    @staticmethod
    def __get_steps(lines):
        all_step = []
        for line in lines:
            if line[0] not in '0123456789':
                continue
            if line in PGNReader.RESULT_STRING_LIST:
                all_step.append(line)
                return all_step
            items = line.split(".")
            steps = items[1].strip().split(" ")
            all_step.append(steps)

        return all_step
