# Generated by Django 4.1.4 on 2022-12-23 02:05

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("board", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="board", old_name="regcnt", new_name="readcnt",
        ),
    ]
