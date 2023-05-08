"""
Module to make reference components table
"""
from typing import ClassVar

from pydantic import Field
from sqlalchemy import VARCHAR, Column, Float, Integer
from sqlalchemy.orm import relationship

from winterdrp.pipelines.reference_building.db_models.basemodel import (
    RefBase,
    dec_field,
    ra_field,
)
from winterdrp.processors.sqldatabase.basemodel import BaseDB


class RefComponentsTable(RefBase):
    """
    Table for individual reference images
    """

    __tablename__ = "refcomponents"

    compid = Column(Integer, primary_key=True)
    query_ra = Column(Float)
    query_dec = Column(Float)

    multiframe_id = Column(Integer)
    extension_id = Column(Integer)
    lx = Column(Integer)
    ly = Column(Integer)
    hx = Column(Integer)
    hy = Column(Integer)
    ra0_0 = Column(Float)
    dec0_0 = Column(Float)
    ra0_1 = Column(Float)
    dec0_1 = Column(Float)
    ra1_0 = Column(Float)
    dec1_0 = Column(Float)
    ra1_1 = Column(Float)
    dec1_1 = Column(Float)
    ra_cent = Column(Float)
    dec_cent = Column(Float)
    savepath = Column(VARCHAR(255))
    query_url = Column(VARCHAR(255))
    ukirt_filename = Column(VARCHAR(255))

    stackcomponents = relationship(
        "RefStackComponentsTable", back_populates="components"
    )


class RefComponents(BaseDB):
    """
    Pydantic model for Reference components
    """

    sql_model: ClassVar = RefComponentsTable

    query_ra: float = ra_field
    query_dec: float = dec_field
    savepath: str = Field(min_length=1)
    query_url: str = Field(min_length=1)
    multiframe_id: int = Field(ge=0)
    extension_id: int = Field(ge=0)
    lx: int = Field(ge=0)
    ly: int = Field(ge=0)
    hx: int = Field(ge=0)
    hy: int = Field(ge=0)
    ra0_0: float = ra_field
    dec0_0: float = dec_field
    ra0_1: float = ra_field
    dec0_1: float = dec_field
    ra1_0: float = ra_field
    dec1_0: float = dec_field
    ra1_1: float = ra_field
    dec1_1: float = dec_field
    ra_cent: float = ra_field
    dec_cent: float = dec_field
    ukirt_filename: str = Field(min_length=1)

    def exists(self) -> bool:
        """
        Checks if the pydantic-ified data exists the corresponding sql database

        :return: bool
        """
        return self.sql_model().exists(
            values=[self.query_ra, self.query_dec], keys=["query_ra", "query_dec"]
        )
