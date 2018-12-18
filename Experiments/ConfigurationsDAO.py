from Experiments import MysqlConnector


class ConfigurationsDAO(object):

    @staticmethod
    def get_configurations(file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations, use_orthogonal_arrays,
                             use_explore_nodes, randomize_remaining_data, random_batch_size, min_oa_strength):
        db = MysqlConnector.get_connection()
        cur = db.cursor()

        query = "SELECT * FROM CONFIGURATIONS WHERE file_name = '%s' AND total_num_of_instances = %s AND possible_gates = '%s' AND subset_min = %s AND subset_max = %s " \
                "AND max_num_of_iterations = %s AND use_orthogonal_arrays = %s AND use_explore_nodes = %s AND randomize_remaining_data = %s " \
                "AND random_batch_size = %s AND min_oa_strength = %s"
        cur.execute(query % (file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations, use_orthogonal_arrays,
                 use_explore_nodes, randomize_remaining_data, random_batch_size, min_oa_strength))

        res = cur.fetchall()

        cur.close()
        db.close()

        return res

    @staticmethod
    def insert_configuration(file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations, use_orthogonal_arrays,
                             use_explore_nodes, randomize_remaining_data, random_batch_size, min_oa_strength):
        db = MysqlConnector.get_connection()
        cur = db.cursor()

        query = "INSERT INTO CONFIGURATIONS(file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations, " \
                "use_orthogonal_arrays, use_explore_nodes, randomize_remaining_data, random_batch_size, min_oa_strength) " \
                "VALUES ('%s', %s, '%s', %s, %s, %s, %s, %s, %s, %s, %s)"
        cur.execute(query % (file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations, use_orthogonal_arrays,
                 use_explore_nodes, randomize_remaining_data, random_batch_size, min_oa_strength))
        last_insert_id = cur._last_insert_id
        db.commit()
        cur.close()
        db.close()
        return last_insert_id
