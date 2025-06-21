# accounts/migrations/0004_update_user_roles.py

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0003_user_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='role',
            field=models.CharField(
                choices=[
                    ('admin', 'Admin'),
                    ('manager', 'Manager'),
                    ('reviewer', 'Reviewer'),
                    ('user', 'User')
                ],
                default='user',
                max_length=20
            ),
        ),
    ]