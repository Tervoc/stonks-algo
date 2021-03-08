import pyodbc 
import os
class InsertDB:
    @staticmethod
    def insert_builder(table, param_dict):
        """
            Description: \n
            SQL insert query tool to assist with inserting in to the database \n

            Parameters: \n
            table (str): Name of the table to insert into \n
            param_dict (param str: param values): Name of parameter and its value in dictionary format \n
        """
        conn = eval(os.environ.get('MALACHI_SERVER'))
        cursor = conn.cursor()

        query = "INSERT INTO " + table + "("

        count = 0

        for key in param_dict.keys():
            if count == 0:
                query += key 
            else: 
                query += ", " + key
            count += 1
        
        query += ") VALUES ("

        for i in range(count):
            if i == 0:
                query += "?" 
            else: 
                query += ", " + "?" 
        query += ")"

        cursor.execute(query, tuple(param_dict.values()))

        conn.commit()
        cursor.close()
        conn.close()
