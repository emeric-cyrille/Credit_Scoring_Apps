# Generated by Django 5.0.4 on 2024-04-18 12:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_review', '0004_alter_model_algorithm'),
    ]

    operations = [
        migrations.AddField(
            model_name='model',
            name='status',
            field=models.CharField(choices=[('trained', 'Entrainé'), ('untrained', 'Non entrainé')], default='Non entrainé', max_length=100),
        ),
        migrations.AlterField(
            model_name='model',
            name='model_data',
            field=models.FileField(default=None, upload_to='models/'),
        ),
    ]
