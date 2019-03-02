from Experiments import MysqlConnector


class IterationsDAO(object):

    @staticmethod
    def get_iterations(iterations_id=None):
        db = MysqlConnector.get_connection()
        cur = db.cursor()

        query = "SELECT * FROM Iterations WHERE id = %s"
        cur.execute(query % (iterations_id))

        res = cur.fetchall()

        cur.close()
        db.close()

        return res

    @staticmethod
    def insert_iteration(experiment_fk, iteration_number, num_of_instances, edges, vertices, component_distribution_and,
                         component_distribution_or,
                         component_distribution_not, component_distribution_xor, degree_distribution, avg_vertex_degree,
                         test_set_error, oa_is_optimal, iteration_time):
        db = MysqlConnector.get_connection()
        cur = db.cursor()

        query = "INSERT INTO ITERATIONS(experiment_fk, iteration_number, num_of_instances, edges, vertices, component_distribution_and, component_distribution_or, " \
                "component_distribution_not, component_distribution_xor, degree_distribution, avg_vertex_degree, test_set_error, oa_is_optimal, iteration_time) " \
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, '%s', %s, %s, %s, %s)"
        cur.execute(query % (experiment_fk, iteration_number, num_of_instances, edges, vertices, component_distribution_and, component_distribution_or,
                         component_distribution_not, component_distribution_xor, degree_distribution, avg_vertex_degree,
                        test_set_error, oa_is_optimal, iteration_time))
        last_insert_id = cur._last_insert_id
        db.commit()
        cur.close()
        db.close()
        return last_insert_id
