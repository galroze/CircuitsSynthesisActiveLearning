from Experiments import MysqlConnector


class ExperimentsDAO(object):

    @staticmethod
    def get_experiments(experiment_id=None):
        db = MysqlConnector.get_connection()
        cur = db.cursor()

        query = "SELECT * FROM EXPERIMENTS WHERE id = %s"
        cur.execute(query % (experiment_id))

        res = cur.fetchall()

        cur.close()
        db.close()

        return res

    @staticmethod
    def insert_experiment(git_version, configuration_fk, run_time):
        db = MysqlConnector.get_connection()
        cur = db.cursor()

        query = "INSERT INTO EXPERIMENTS(git_version, configuration_fk, run_time) VALUES ('%s', %s, %s)"
        cur.execute(query % (git_version, configuration_fk, run_time))

        last_insert_id = cur._last_insert_id

        db.commit()
        cur.close()
        db.close()

        return last_insert_id
