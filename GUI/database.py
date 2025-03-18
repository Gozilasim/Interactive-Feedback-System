import mysql.connector

def initialize_mysql():
    # Connect to MySQL database
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='survey_data'  
    )
    return connection

def add_data_to_database(database):
    data = {
        "gender" : database[0],
        "age" : database[1],
        "Q1": database[2],
        "Q2": database[3],
        "Q3": database[4],
        "Q4": database[5],
        "Q5": database[6]
    }
    
    # Initialize MySQL connection
    connection = initialize_mysql()
    cursor = connection.cursor()

    # Insert data into MySQL table
    insert_query = '''
    INSERT INTO survey (gender,age,Q1, Q2, Q3, Q4, Q5)
    VALUES (%s,%s,%s, %s, %s, %s, %s)
    '''
    
    try:
      cursor.execute(insert_query, (data["gender"], data["age"], data["Q1"], data["Q2"], data["Q3"], data["Q4"], data["Q5"]))
      connection.commit()
    except mysql.connector.Error as err:
      print(f"Error: {err}")
    finally:
      cursor.close()
      connection.close()


    print("Data added to MySQL:", data)

    cursor.close()
    connection.close()

