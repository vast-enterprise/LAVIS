from os import environ

DB_HOST = environ.get("DB_HOST")
DB_USER = environ.get("DB_USER")
DB_PORT = environ.get("DB_PORT")
DB_PASSWORD = environ.get("DB_PASSWORD")
DB_DATABASE = environ.get("DB_DATABASE")
DB_CAPTION_TABLE = environ.get("DB_CAPTION_TABLE", "model_image_caption")
DB_IMAGE_TABLE = environ.get("DB_IMAGE_TABLE", "model_image")

error_list = []
if DB_HOST is None:
    error_list.append("DB_HOST")
if DB_USER is None:
    error_list.append("DB_USER")
if DB_PORT is None:
    error_list.append("DB_PORT")
if DB_PASSWORD is None:
    error_list.append("DB_PASSWORD")
if DB_DATABASE is None:
    error_list.append("DB_DATABASE")
assert len(error_list) == 0, "Please set the following environment variables: {}".format(", ".join(error_list))

import pymysql

connection = None
def get_database_connection():
    # TODO: 换成连接池
    global DB_HOST, DB_USER, DB_PORT, DB_PASSWORD, DB_DATABASE, connection
    if connection is None:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            port=int(DB_PORT),
            password=DB_PASSWORD,
            database=DB_DATABASE
        )
    else:
        # test the connection is alive
        try:
            connection.ping(reconnect=True)
        except Exception as e:
            connection = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                port=int(DB_PORT),
                password=DB_PASSWORD,
                database=DB_DATABASE
            )
            raise
    return connection

def write_image_caption_to_database(store_caption_list):
    """
    store_caption_list: [(image_id, image_embedding_uri, image_caption, image_caption_embedding_uri)]
    """
    # write dict into mysql database
    # make new connection to database:
    conn = get_database_connection()
    conn.begin()
    cursor = conn.cursor()
    select_sql = f"SELECT image_id, cap_model_tag, extra, \
            image_embedding_bucket, image_embedding_uri, image_embedding_shape, image_caption, \
            image_caption_embedding_bucket, image_caption_embedding_uri, image_caption_embedding_shape \
            FROM `{DB_DATABASE}`.`{DB_CAPTION_TABLE}` WHERE image_id = '%s' and is_delete = 0"
    insert_sql = f"INSERT INTO `{DB_DATABASE}`.`{DB_CAPTION_TABLE}` (image_id, cap_model_tag, extra, \
        image_embedding_bucket, image_embedding_uri, image_embedding_shape, image_caption, \
        image_caption_embedding_bucket, image_caption_embedding_uri, image_caption_embedding_shape) \
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')"
    update_sql = "UPDATE `" + DB_DATABASE + "`.`" + DB_CAPTION_TABLE + "` SET cap_model_tag = '{1}', extra = '{2}', \
        image_embedding_bucket = '{3}', image_embedding_uri = '{4}', image_embedding_shape = '{5}', image_caption = '{6}', \
        image_caption_embedding_bucket = '{7}', image_caption_embedding_uri = '{8}', image_caption_embedding_shape = '{9}' \
        WHERE image_id = '{0}' and is_delete = 0"
    try:
        insert_cnt, update_cnt, skip_cnt = 0, 0, 0
        for image_id, image_embedding_uri, image_caption, image_caption_embedding_uri in store_caption_list:
            cursor.execute(select_sql % image_id)
            record = cursor.fetchall()
            assert len(record) == 0 or len(record) == 1, "valid image_id should be unique"
            # TODO: 用配置文件管理固定的值
            tmp = (image_id, "LAVIS-blip2_t5-xxl,CLIP-ViT-B/32", '{}', 'image_embedding_bucket',\
                                image_embedding_uri, '(32, 256)', image_caption.replace("'", "\\'").replace('"', '\\"'), \
                                'image_caption_embedding_bucket', image_caption_embedding_uri, '(256,)')
            if len(record) == 1:
                if record[0][0:10] == tmp:
                    print(f"image_id {image_id} already exists in database, and there is no change, so skip...")
                    skip_cnt += 1
                else:
                    print(f"image_id {image_id} already exists in database, but there are changes, so update...")
                    cursor.execute(update_sql.format(*tmp))
                    update_cnt += 1
            else:
                tmp = insert_sql % tmp
                cursor.execute(tmp)
                insert_cnt += 1
        conn.commit()
        print(f"write to database successfully, insert {insert_cnt} rows, update {update_cnt} rows, skip {skip_cnt} rows")
    except Exception:
        conn.rollback()
        print("fail to write to database")
        raise
    finally:
        cursor.close()

def get_image_paths_from_model_id(model_id):
    conn = get_database_connection()
    # FIXME: 选择8个角上的图片
    name_like = [f"name LIKE 'render_00{i:02d}%'" for i in range(6, 14)]
    name_like = " OR ".join(name_like)
    name_like = ' AND (' + name_like + ')'
    select_sql = f"SELECT id, pfs_path FROM `{DB_DATABASE}`.`{DB_IMAGE_TABLE}` WHERE model_id = {model_id} and is_delete = 0 and type = 'render'" + name_like
    try:
        cursor = conn.cursor()
        cursor.execute(select_sql)
        return cursor.fetchall()
    except Exception:
        print("fail to read from database")
        raise
    finally:
        cursor.close()


if __name__ == '__main__':
    select_sql = f"SELECT image_caption FROM `{DB_DATABASE}`.`{DB_CAPTION_TABLE}` WHERE image_id = '%s' and is_delete = 0"
    conn = get_database_connection()
    for i in range(1, 11):
        print('\n'*5)
        print(f"获取第{i}个模型的caption")
        path_list = get_image_paths_from_model_id(i)
        print(path_list[0][1])
        print('\n')
        caption_list = []
        for image_id, pfs_path in path_list:
            cursor = conn.cursor()
            cursor.execute(select_sql % image_id)
            records = cursor.fetchall()
            assert len(records) == 1
            caption_list.append(records[0][0])
        print(". ".join(caption_list))
