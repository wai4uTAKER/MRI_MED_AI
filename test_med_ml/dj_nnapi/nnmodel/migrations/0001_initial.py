

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AuthGroup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150, unique=True)),
            ],
            options={
                'db_table': 'auth_group',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthGroupPermissions',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'auth_group_permissions',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthPermission',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('codename', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'auth_permission',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoAdminLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('action_time', models.DateTimeField()),
                ('object_id', models.TextField(blank=True, null=True)),
                ('object_repr', models.CharField(max_length=200)),
                ('action_flag', models.SmallIntegerField()),
                ('change_message', models.TextField()),
            ],
            options={
                'db_table': 'django_admin_log',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoContentType',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('app_label', models.CharField(max_length=100)),
                ('model', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'django_content_type',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoMigrations',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('app', models.CharField(max_length=255)),
                ('name', models.CharField(max_length=255)),
                ('applied', models.DateTimeField()),
            ],
            options={
                'db_table': 'django_migrations',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoSession',
            fields=[
                ('session_key', models.CharField(max_length=40, primary_key=True, serialize=False)),
                ('session_data', models.TextField()),
                ('expire_date', models.DateTimeField()),
            ],
            options={
                'db_table': 'django_session',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='InnerMailMaildetails',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('msg', models.TextField()),
                ('mail_type', models.IntegerField()),
                ('nodule_type', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'inner_mail_maildetails',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='InnerMailNotification',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('create_date', models.DateTimeField()),
            ],
            options={
                'db_table': 'inner_mail_notification',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='InnerMailNotificationdynamics',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('status', models.IntegerField()),
                ('update_date', models.DateTimeField()),
            ],
            options={
                'db_table': 'inner_mail_notificationdynamics',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='InnerMailNotificationgroup',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=512)),
                ('create_date', models.DateTimeField()),
            ],
            options={
                'db_table': 'inner_mail_notificationgroup',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='InnerMailNotificationgroupMembers',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'inner_mail_notificationgroup_members',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlMedworker',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('password', models.CharField(max_length=128)),
                ('last_login', models.DateTimeField(blank=True, null=True)),
                ('is_superuser', models.BooleanField()),
                ('first_name', models.CharField(max_length=150)),
                ('last_name', models.CharField(max_length=150)),
                ('is_staff', models.BooleanField()),
                ('is_active', models.BooleanField()),
                ('date_joined', models.DateTimeField()),
                ('email', models.CharField(max_length=254, unique=True)),
                ('is_remote_worker', models.BooleanField()),
                ('fathers_name', models.CharField(max_length=150)),
                ('med_organization', models.CharField(max_length=512)),
                ('job', models.CharField(blank=True, max_length=256, null=True)),
                ('expert_details', models.TextField()),
            ],
            options={
                'db_table': 'med_auth_medworker',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlMedworkerGroups',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'medml_medworker_groups',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlMedworkerUserPermissions',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'medml_medworker_user_permissions',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlMlmodel',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=256)),
                ('file', models.CharField(max_length=100)),
                ('model_type', models.CharField(max_length=1)),
                ('projection_type', models.CharField(max_length=10)),
            ],
            options={
                'db_table': 'medml_mlmodel',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlPatient',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('first_name', models.CharField(max_length=150)),
                ('last_name', models.CharField(max_length=150)),
                ('fathers_name', models.CharField(max_length=150)),
                ('personal_policy', models.CharField(max_length=16)),
                ('email', models.CharField(max_length=254, unique=True)),
                ('is_active', models.BooleanField()),
            ],
            options={
                'db_table': 'med_auth_patient',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlPatientcard',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('acceptance_datetime', models.DateTimeField()),
                ('has_nodules', models.CharField(max_length=128)),
                ('diagnosis', models.TextField()),
            ],
            options={
                'db_table': 'medml_patientcard',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlUzidevice',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=512)),
            ],
            options={
                'db_table': 'medml_uzidevice',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MedmlUziimage',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('brightness', models.FloatField()),
                ('contrast', models.FloatField()),
                ('sharpness', models.FloatField()),
                ('image_count', models.IntegerField()),
                ('details', models.JSONField()),
                ('diagnos_date', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'medml_uziimage',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='OriginalImage',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('create_date', models.DateTimeField()),
                ('delay_time', models.FloatField()),
                ('viewed_flag', models.BooleanField()),
                ('image', models.CharField(max_length=100)),
            ],
            options={
                'verbose_name': 'Снимок оригинала',
                'verbose_name_plural': 'Снимки оригиналов',
                'db_table': 'medml_originalimage',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='SegmentationData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('details', models.JSONField(null=True)),
            ],
            options={
                'verbose_name': 'Сегмент',
                'verbose_name_plural': 'Сегменты',
                'db_table': 'nnmodel_segmentationdata',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='SegmentationPoint',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.BigIntegerField()),
                ('x', models.PositiveIntegerField()),
                ('y', models.PositiveIntegerField()),
                ('z', models.PositiveIntegerField(default=0)),
            ],
            options={
                'verbose_name': 'Точка сегмента',
                'verbose_name_plural': 'Точки сегментов',
                'db_table': 'nnmodel_segmentationpoint',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='UZISegmentGroupInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('details', models.JSONField(null=True)),
                ('is_ai', models.BooleanField(default=False)),
            ],
            options={
                'verbose_name': 'Информация о группе сегментов',
                'verbose_name_plural': 'Информация о группе сегментов',
                'db_table': 'nnmodel_uzisegmentgroupinfo',
                'managed': False,
            },
        ),
    ]
