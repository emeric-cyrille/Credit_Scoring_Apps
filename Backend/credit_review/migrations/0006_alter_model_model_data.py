# Generated by Django 5.0.4 on 2024-04-18 12:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_review', '0005_model_status_alter_model_model_data'),
    ]

    operations = [
        migrations.AlterField(
            model_name='model',
            name='model_data',
            field=models.FileField(blank=True, null=True, upload_to='models/'),
        ),
    ]